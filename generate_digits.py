import torch
import torch.nn as nn # Wird für DenoiseModel Definition benötigt
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Konfiguration (sollte mit training.py übereinstimmen) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 28
TIMESTEPS = 200 # Muss mit dem Wert aus dem Training übereinstimmen
MODEL_SAVE_PATH = "ddpm_mnist.pth"
RESULTS_FOLDER = "results_generation"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- Diffusionszeitplan (wird für den Sampling-Prozess benötigt) ---
# Diese Werte müssen exakt denen aus training.py entsprechen!
BETA_START = 1e-4
BETA_END = 0.02
BETAS = torch.linspace(BETA_START, BETA_END, TIMESTEPS, device=DEVICE)
ALPHAS = 1. - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, axis=0)
# SQRT_RECIP_ALPHAS = torch.sqrt(1.0 / ALPHAS) # Nicht direkt für dieses einfache Sampling verwendet
# SQRT_ONE_MINUS_ALPHAS_CUMPROD = torch.sqrt(1. - ALPHAS_CUMPROD) # Wird indirekt in p_sample benötigt
# POSTERIOR_VARIANCE = BETAS * (1. - torch.cat([torch.tensor([1.0-BETA_START], device=DEVICE), ALPHAS_CUMPROD[:-1]])) / (1. - ALPHAS_CUMPROD)

# Die Hilfsfunktion 'extract' muss hier auch definiert sein, wenn sie in p_sample verwendet wird.
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# --- DenoiseModel Definition (exakt wie in training.py) ---
# Es ist wichtig, dass die Modellarchitektur hier identisch ist,
# damit die gelernten Gewichte korrekt geladen werden können.
class DenoiseModel(nn.Module):
    def __init__(self, img_size, timesteps):
        super().__init__()
        self.img_size = img_size
        input_dim = img_size * img_size

        self.time_mlp = nn.Sequential(
            nn.Linear(timesteps, timesteps * 4),
            nn.SiLU(),
            nn.Linear(timesteps * 4, timesteps)
        )

        self.model = nn.Sequential(
            nn.Linear(input_dim + timesteps, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, input_dim),
            nn.Tanh()
        )

    def forward(self, x, t):
        x = x.view(x.shape[0], -1)
        t_emb = self.time_mlp(self.time_embedding(t, TIMESTEPS))
        x_t_emb = torch.cat((x, t_emb), dim=-1)
        return self.model(x_t_emb).view(x.shape[0], 1, self.img_size, self.img_size)

    def time_embedding(self, t, channels):
        embedding = torch.zeros(t.shape[0], channels, device=t.device)
        embedding[torch.arange(t.shape[0]), t.long()] = 1
        return embedding

# --- Sampling-Funktion (p_sample_loop aus DDPM) ---
@torch.no_grad() # Wichtig: Keine Gradientenberechnung beim Sampling
def p_sample(model, x, t, t_index):
    """
    Einzelner Schritt des Umkehrdiffusionsprozesses (Entrauschen).
    """
    betas_t = extract(BETAS, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(torch.sqrt(1. - ALPHAS_CUMPROD), t, x.shape)
    sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / ALPHAS), t, x.shape)

    # Gleichung 11 aus DDPM: x_t-1 = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
    # Vorhersage des Rauschens durch das Modell
    predicted_noise = model(x, t)

    # Mittelwert der Verteilung für x_{t-1}
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean # Kein Rauschen im letzten Schritt hinzufügen
    else:
        # Rauschen für den Sampling-Schritt hinzufügen
        posterior_variance_t = extract(BETAS, t, x.shape) # Vereinfachung: sigma_t^2 = beta_t
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def generate_samples(model, num_samples=16, timesteps=TIMESTEPS, device=DEVICE, img_size=IMG_SIZE):
    """
    Generiert neue Bilder mit dem trainierten Modell.
    """
    print(f"Generating {num_samples} samples...")
    model.eval() # Modell in den Evaluationsmodus setzen

    # Start mit zufälligem Rauschen (x_T)
    img = torch.randn((num_samples, 1, img_size, img_size), device=device)

    for i in reversed(range(0, timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i)
        if i % (timesteps//10) == 0 :
             print(f"Sampling step {i}/{timesteps}")

    # Denormalisieren von [-1,1] zu [0,1]
    img = (img + 1) / 2
    img = torch.clamp(img, 0.0, 1.0) # Werte auf [0,1] beschränken
    return img

def save_generated_images(images, folder, base_filename="generated_digit"):
    """Speichert generierte Bilder einzeln und als Grid."""
    images_np = images.cpu().numpy()

    # Grid-Bild speichern
    fig_grid, axes_grid = plt.subplots(int(np.sqrt(images.size(0))), int(np.sqrt(images.size(0))), figsize=(8,8))
    axes_grid = axes_grid.flatten()
    for i, ax in enumerate(axes_grid):
        ax.imshow(images_np[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    grid_path = os.path.join(folder, f"{base_filename}_grid.png")
    plt.savefig(grid_path)
    print(f"Saved grid of generated images to {grid_path}")
    plt.close(fig_grid)

    # Einzelne Bilder speichern (optional)
    # for i in range(images.size(0)):
    #     plt.figure(figsize=(2,2))
    #     plt.imshow(images_np[i].squeeze(), cmap='gray')
    #     plt.axis('off')
    #     individual_path = os.path.join(folder, f"{base_filename}_{i+1}.png")
    #     plt.savefig(individual_path)
    #     plt.close()
    # print(f"Saved {images.size(0)} individual images to {folder}")


if __name__ == '__main__':
    # Modell laden
    model = DenoiseModel(img_size=IMG_SIZE, timesteps=TIMESTEPS).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"Model loaded from {MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Please train the model first using training.py.")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Neue Ziffern generieren
    num_new_digits = 16 # z.B. 4x4 Grid
    generated_images = generate_samples(model, num_samples=num_new_digits, timesteps=TIMESTEPS, device=DEVICE, img_size=IMG_SIZE)

    # Generierte Bilder speichern
    save_generated_images(generated_images, RESULTS_FOLDER)