#app.py

import time
import random
import os

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.ndimage import gaussian_filter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


# 页面与基础配置
st.set_page_config(page_title="A8", layout="wide")
st.title("A8")

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 2


# 模型架构
class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 784)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


@st.cache_resource
def init_models():
    """缓存模型加载，防止每次交互都重新读取硬盘"""
    m_ae = AutoEncoder(LATENT_DIM).to(device)
    m_vae = VAE(LATENT_DIM).to(device)
    m_gen = Generator().to(device)
    m_disc = Discriminator().to(device)

    # 尝试静默加载权重（如果没有也不会报错崩溃）
    def load_safe(model, path):
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            return True
        except:
            return False

    status = {
        "AE": load_safe(m_ae, "checkpoints/ae.pth"),
        "VAE": load_safe(m_vae, "checkpoints/vae.pth"),
        "Generator": load_safe(m_gen, "checkpoints/generator.pth"),
        "Discriminator": load_safe(m_disc, "checkpoints/discriminator.pth")
    }
    
    m_ae.eval(); m_vae.eval(); m_gen.eval(); m_disc.eval()
    return m_ae, m_vae, m_gen, m_disc, status

@st.cache_data
def load_and_process_data():
    """缓存数据处理，直接固定使用 MNIST 数据集"""
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # 获取固定测试样本 (8张包含数字7的图)
    sample_imgs, sample_lbls = [], []
    for img, label in dataset:
        if int(label) == 7:
            sample_imgs.append(img)
            sample_lbls.append(label)
        if len(sample_imgs) == 8: break
    sample_images = torch.stack(sample_imgs)
    flat_images = sample_images.view(-1, 784).to(device)

    # 获取插值样本 (找一张2，找一张7)
    img_2, img_7 = None, None
    for img, label in dataset:
        if int(label) == 2 and img_2 is None: img_2 = img.view(1, -1).to(device)
        if int(label) == 7 and img_7 is None: img_7 = img.view(1, -1).to(device)
        if img_2 is not None and img_7 is not None: break

    # 获取潜空间坐标 (采样 2000 个点)
    loader = DataLoader(dataset, batch_size=200, shuffle=True)
    vectors, labels = [], []
    _, temp_vae, _, _, _ = init_models()
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            if i >= 10: break # 10 batch * 200 = 2000 个点
            mu, _ = temp_vae.encode(imgs.view(-1, 784).to(device))
            vectors.append(mu.cpu().numpy())
            labels.append(lbls.numpy())
            
    latent_df = pd.DataFrame({
        "z1": np.concatenate(vectors)[:, 0],
        "z2": np.concatenate(vectors)[:, 1],
        "label": np.concatenate(labels).astype(str)
    })

    return dataset, sample_images, flat_images, img_2, img_7, latent_df


# 初始化应用状态
st.sidebar.title("实验配置")

# 加载数据与模型
ae, vae, generator, discriminator, load_status = init_models()
dataset, sample_images, flat_images, img_2, img_7, latent_df = load_and_process_data()

st.sidebar.markdown("### 模型加载状态")
for name, is_loaded in load_status.items():
    if is_loaded: st.sidebar.success(f" {name} Loaded")
    else: st.sidebar.warning(f"{name} (Random Weights)")

# 预计算重构结果
with torch.no_grad():
    ae_recon = ae(flat_images).cpu()
    vae_recon, _, _ = vae(flat_images)
    vae_recon = vae_recon.cpu()


tabs = st.tabs(["AE/VAE 重构对比", "潜空间探索与插值", "GAN / Diffusion 实验"])

# ----------------- TAB 1: 重构对比 -----------------
with tabs[0]:
    st.header("AutoEncoder 与 VAE 重构对比")
    c1, c2, c3 = st.columns(3)

    def plot_images(col, title, images):
        col.subheader(title)
        fig, axes = plt.subplots(1, 8, figsize=(12, 2))
        for i in range(8):
            axes[i].imshow(images[i].reshape(28, 28), cmap="gray")
            axes[i].axis("off")
        col.pyplot(fig)

    plot_images(c1, "Original Images", sample_images.numpy())
    plot_images(c2, "AE Reconstruction", ae_recon.numpy())
    plot_images(c3, "VAE Reconstruction", vae_recon.numpy())

    st.subheader("Reconstruction Error Heatmap")
    err_ae = torch.abs(flat_images.cpu() - ae_recon)[0].reshape(28, 28)
    err_vae = torch.abs(flat_images.cpu() - vae_recon)[0].reshape(28, 28)

    e1, e2 = st.columns(2)
    with e1:
        fig, ax = plt.subplots()
        im = ax.imshow(err_ae, cmap="hot"); ax.set_title("AE Error")
        plt.colorbar(im); st.pyplot(fig)
    with e2:
        fig, ax = plt.subplots()
        im = ax.imshow(err_vae, cmap="hot"); ax.set_title("VAE Error")
        plt.colorbar(im); st.pyplot(fig)

    st.subheader("📉 Training Convergence (Simulated)")
    epochs = np.arange(1, 21)
    ae_loss = 0.12 * np.exp(-epochs / 7) + np.random.normal(0, 0.002, 20)
    vae_loss = 0.18 * np.exp(-epochs / 10) + np.random.normal(0, 0.003, 20)
    st.line_chart(pd.DataFrame({"AE Loss": ae_loss, "VAE Loss": vae_loss}, index=epochs))

# ----------------- TAB 2: 潜空间探索 -----------------
with tabs[1]:
    st.header("VAE 潜空间探索")
    
    col_scatter, col_gen = st.columns([2, 1])

    with col_scatter:
        fig = px.scatter(latent_df, x="z1", y="z2", color="label", 
                         title="2D Latent Space (直接点击图中的点进行生成)")
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

    with col_gen:
        st.subheader("生成结果")
        if event and len(event.selection["points"]) > 0:
            z1 = event.selection["points"][0]["x"]
            z2 = event.selection["points"][0]["y"]
            st.success(f"坐标: ({z1:.2f}, {z2:.2f})")
        else:
            z1, z2 = 0.0, 0.0
            st.info("请点击左侧散点图。当前显示原点 (0,0)")

        z_tensor = torch.tensor([[z1, z2]], dtype=torch.float32).to(device)
        with torch.no_grad():
            generated = vae.decode(z_tensor).cpu()

        fig_gen, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(generated[0].reshape(28, 28), cmap="gray")
        ax.axis("off")
        st.pyplot(fig_gen)

    st.divider()
    st.subheader("潜空间插值 (Digit 2 ➔ 7)")

    if img_2 is not None and img_7 is not None:
        with torch.no_grad():
            mu1, _ = vae.encode(img_2)
            mu2, _ = vae.encode(img_7)

        steps_interp = st.slider("Interpolation Steps", 5, 15, 10)
        interpolation_results = []
        for alpha in np.linspace(0, 1, steps_interp):
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            with torch.no_grad():
                recon = vae.decode(z_interp).cpu()
            interpolation_results.append(recon.numpy())

        fig, axes = plt.subplots(1, steps_interp, figsize=(15, 2))
        for i in range(steps_interp):
            axes[i].imshow(interpolation_results[i][0].reshape(28, 28), cmap="gray")
            axes[i].axis("off")
        st.pyplot(fig)

# ----------------- TAB 3: GAN & Diffusion -----------------
with tabs[2]:
    st.header("GAN / Diffusion 实验")

    st.subheader("🎲 DCGAN Sample Generation")
    gan_seed = st.number_input("GAN Random Seed", value=42, step=1)
    np.random.seed(int(gan_seed)); torch.manual_seed(int(gan_seed))

    noise_scale = st.slider("Noise Scale", 0.1, 3.0, 1.0, 0.1)
    z_noise = noise_scale * torch.randn(64, 100).to(device)
    
    st.write("### Generator Input Noise (First 100 dims)")
    st.line_chart(z_noise[0].cpu().numpy(), height=150)

    with torch.no_grad():
        fake_images = generator(z_noise)
        real_score = discriminator(flat_images).mean()
        fake_score = discriminator(fake_images).mean()

    c1, c2 = st.columns(2)
    c1.metric("Discriminator Real Score (Target 1.0)", f"{real_score.item():.4f}")
    c2.metric("Discriminator Fake Score (Target 0.0)", f"{fake_score.item():.4f}")

    grid = make_grid(fake_images.reshape(-1, 1, 28, 28), nrow=8, normalize=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    ax.axis("off")
    st.pyplot(fig)

    st.divider()
    st.subheader("Diffusion Prompt 实验 (Simulated)")
    
    d1, d2 = st.columns([2, 1])
    with d1:
        prompt = st.text_input("Prompt", "A neon glowing handwritten number")
        negative_prompt = st.text_input("Negative Prompt", "blurry, dark")
    with d2:
        steps = st.slider("Sampling Steps", 1, 50, 20)
        guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)

    if st.button("Generate Diffusion Image"):
        with st.status("Running Denoising Steps...") as status:
            time.sleep(0.5)
            status.update(label="Encoding Prompt...", state="running")
            time.sleep(0.5)
            status.update(label="Applying Guidance...", state="running")
            time.sleep(0.5)
            status.update(label="Generation Complete!", state="complete")

        base_img = sample_images[0].squeeze().numpy()
        compare_imgs = []
        guidance_list = [guidance / 2, guidance, guidance * 1.5]

        for g in guidance_list:
            smooth = gaussian_filter(base_img, sigma=max(0.5, 8 / g))
            noise = np.random.normal(0, 1 / (g + 1), (28, 28))
            compare_imgs.append(np.clip(smooth + noise, 0, 1))

        st.write("### 不同 Guidance Scale 对比")
        compare_fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for i in range(3):
            axes[i].imshow(compare_imgs[i], cmap="gray")
            axes[i].set_title(f"Guidance = {guidance_list[i]:.1f}")
            axes[i].axis("off")
        st.pyplot(compare_fig)
