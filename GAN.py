import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成器模型
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 训练 GAN 模型
def train_gan(generator, discriminator, gan, dataloader, num_epochs=10000, latent_dim=10, lr=0.0002):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for real_data in dataloader:
            # 训练判别器
            optimizer_d.zero_grad()
            real_labels = torch.ones(real_data.size(0), 1) #真实数据
            fake_labels = torch.zeros(real_data.size(0), 1) #生成数据

            real_output = discriminator(real_data)
            real_loss = criterion(real_output, real_labels)
            real_loss.backward()

            z = torch.randn(real_data.size(0), latent_dim)
            fake_data = generator(z)
            fake_output = discriminator(fake_data.detach())
            fake_loss = criterion(fake_output, fake_labels)
            fake_loss.backward()

            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            fake_output = discriminator(fake_data)
            gen_loss = criterion(fake_output, real_labels)
            gen_loss.backward()
            optimizer_g.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {real_loss.item() + fake_loss.item()}, G Loss: {gen_loss.item()}")

# 定义并训练 GAN 模型
latent_dim = 10 # 随机噪声的维度
output_dim = 1  # 一维数据
generator = Generator(latent_dim, output_dim)#生成器
discriminator = Discriminator(output_dim)#判别器
gan = nn.Sequential(generator, discriminator)

# 生成一些随机正弦波形数据作为训练集
data = np.sin(np.linspace(0, 4 * np.pi, 1000)).reshape(-1, 1).astype(np.float32)
dataloader = torch.utils.data.DataLoader(torch.from_numpy(data), batch_size=64, shuffle=True)

# 训练 GAN 模型
train_gan(generator, discriminator, gan, dataloader)
