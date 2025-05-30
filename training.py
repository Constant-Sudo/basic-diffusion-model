import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# --- Hyperparameter und Konfiguration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 100 # Kann für bessere Ergebnisse erhöht werden
LR = 1e-3
TIMESTEPS = 200 # Anzahl der Diffusionsschritte
MODEL_SAVE_PATH = "ddpm_mnist.pth"
RESULTS_FOLDER = "results_training"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# --- Diffusionszeitplan (linear) ---
BETA_START = 1e-4
BETA_END = 0.02
BETAS = torch.linspace(BETA_START, BETA_END, TIMESTEPS, device=DEVICE)
ALPHAS = 1. - BETAS
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, axis=0)
ALPHAS_CUMPROD_PREV = torch.cat([torch.tensor([1.0], device=DEVICE), ALPHAS_CUMPROD[:-1]])
SQRT_ALPHAS_CUMPROD = torch.sqrt(ALPHAS_CUMPROD)
SQRT_ONE_MINUS_ALPHAS_CUMPROD = torch.sqrt(1. - ALPHAS_CUMPROD)
POSTERIOR_VARIANCE = BETAS * (1. - ALPHAS_CUMPROD_PREV) / (1. - ALPHAS_CUMPROD)


# --- Hilfsfunktionen für den Diffusionsprozess ---
def extract(a, t, x_shape):
    """Extrahiert die entsprechenden Werte aus 'a' für einen Batch von Zeitstempeln 't'."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(x_start, t, noise=None):
    """
    Vorwärtsdiffusionsprozess: Fügt einem Bild Rauschen hinzu.
    x_start: Originalbild (Batch)
    t: Zeitstempel (Batch)
    noise: Optionales Rauschen, sonst wird zufälliges Rauschen generiert
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(SQRT_ALPHAS_CUMPROD, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(SQRT_ONE_MINUS_ALPHAS_CUMPROD, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# --- Einfaches Entrauschungsnetzwerk (MLP) ---
# Für bessere Ergebnisse wäre hier ein U-Net geeigneter, aber MLP ist einfacher.
class DenoiseModel(nn.Module):
    def __init__(self, img_size, timesteps):
        super().__init__()
        self.img_size = img_size
        input_dim = img_size * img_size

        # Zeit-Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(timesteps, timesteps * 4),
            nn.SiLU(), # Swish/SiLU Aktivierung
            nn.Linear(timesteps * 4, timesteps)
        )

        # Hauptnetzwerk
        self.model = nn.Sequential(
            nn.Linear(input_dim + timesteps, 1024), # Eingabe + Zeit-Embedding
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, input_dim),
            nn.Tanh() # Ausgabe auf [-1, 1] skalieren, passend zur Datennormalisierung
        )

    def forward(self, x, t):
        x = x.view(x.shape[0], -1) # Flatten image
        t_emb = self.time_mlp(self.time_embedding(t, TIMESTEPS))
        x_t_emb = torch.cat((x, t_emb), dim=-1)
        return self.model(x_t_emb).view(x.shape[0], 1, self.img_size, self.img_size)

    def time_embedding(self, t, channels):
        # Erstellt ein einfaches One-Hot-ähnliches Embedding für die Zeit
        # Eine bessere Methode wäre die Verwendung von sinusförmigen Positions-Embeddings
        embedding = torch.zeros(t.shape[0], channels, device=t.device)
        embedding[torch.arange(t.shape[0]), t.long()] = 1
        return embedding

# --- Daten laden und vorbereiten ---
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),                # Konvertiert zu Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalisiert auf [-1, 1]
    ])
    dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    return dataloader

# --- Trainingsfunktion ---
def train(model, dataloader, optimizer, loss_fn, epochs):
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (images, _) in enumerate(dataloader):
            optimizer.zero_grad()

            images = images.to(DEVICE)
            # Zufällige Zeitstempel für den Batch generieren
            t = torch.randint(0, TIMESTEPS, (images.shape[0],), device=DEVICE).long()

            # Rauschen generieren und verrauschte Bilder erstellen
            noise = torch.randn_like(images)
            x_t = q_sample(images, t, noise)

            # Rauschen vorhersagen
            predicted_noise = model(x_t, t)

            # Verlust berechnen
            loss = loss_fn(noise, predicted_noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")

        # Optional: Generiere ein paar Beispielbilder während des Trainings
        if (epoch + 1) % 10 == 0 or epoch == epochs -1 :
            from generate_digits import generate_samples # Importiere hier, um zirkuläre Abhängigkeiten zu vermeiden
            sample_images = generate_samples(model, num_samples=4, timesteps=TIMESTEPS, device=DEVICE, img_size=IMG_SIZE)
            save_image_grid(sample_images, os.path.join(RESULTS_FOLDER, f"epoch_{epoch+1}.png"))
            print(f"Saved sample images for epoch {epoch+1}")


    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

def save_image_grid(images, path, grid_size=(2, 2)):
    """Speichert einen Batch von Bildern als Grid."""
    import matplotlib.pyplot as plt
    images = images.cpu().numpy()
    images = (images + 1) / 2 # Denormalisieren von [-1,1] zu [0,1]
    images = np.clip(images, 0, 1)

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(5,5))
    axes = axes.flatten()
    for img, ax in zip(images, axes):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    dataloader = get_data()
    model = DenoiseModel(img_size=IMG_SIZE, timesteps=TIMESTEPS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss() # Mean Squared Error zwischen tatsächlichem und vorhergesagtem Rauschen

    train(model, dataloader, optimizer, loss_fn, EPOCHS)