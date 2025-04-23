import os
import glob
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class VideoFramesDataset(Dataset):
    def __init__(self, root_dir, frames_per_clip=5, step_between_clips=5, transform=None):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.transform = transform
        self.samples = []
        self._extrair_frames_se_preciso()
        self._preparar_clipes()

    def _extrair_frames_se_preciso(self):
        for classe in os.listdir(self.root_dir):
            caminho_classe = os.path.join(self.root_dir, classe)
            for arquivo in os.listdir(caminho_classe):
                if not arquivo.endswith(".avi"):
                    continue

                caminho_video = os.path.join(caminho_classe, arquivo)
                nome_base = os.path.splitext(arquivo)[0]
                pasta_saida = os.path.join(caminho_classe, nome_base)

                if os.path.exists(pasta_saida) and len(os.listdir(pasta_saida)) > 0:
                    continue  # Já extraído

                os.makedirs(pasta_saida, exist_ok=True)

                cap = cv2.VideoCapture(caminho_video)
                frame_id = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    caminho_frame = os.path.join(pasta_saida, f"frame_{frame_id:04d}.jpg")
                    cv2.imwrite(caminho_frame, frame)
                    frame_id += 1
                cap.release()

    def _preparar_clipes(self):
        for classe in os.listdir(self.root_dir):
            caminho_classe = os.path.join(self.root_dir, classe)
            for pasta_video in os.listdir(caminho_classe):
                caminho_video = os.path.join(caminho_classe, pasta_video)
                if not os.path.isdir(caminho_video):
                    continue

                frames = sorted(glob.glob(os.path.join(caminho_video, '*.jpg')))
                total = len(frames)

                for i in range(0, total - self.frames_per_clip + 1, self.step_between_clips):
                    clipe = frames[i:i + self.frames_per_clip]
                    if len(clipe) == self.frames_per_clip:
                        self.samples.append(clipe)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clipe_paths = self.samples[idx]
        clipe = [Image.open(p).convert('RGB') for p in clipe_paths]
        if self.transform:
            clipe = [self.transform(img) for img in clipe]
        clipe = torch.stack(clipe)  # [T, C, H, W]
        return clipe

class Loader:
    def __init__(self):
        self.transformar = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def load(self):
        dataset = VideoFramesDataset(
            root_dir="./data/UCF-101",
            frames_per_clip=5,
            step_between_clips=5,
            transform=self.transformar
        )

        tamanho_treino = int(0.8 * len(dataset))
        tamanho_teste = len(dataset) - tamanho_treino
        treino, teste = torch.utils.data.random_split(dataset, [tamanho_treino, tamanho_teste])

        loader_treino = DataLoader(treino, batch_size=4, shuffle=True, num_workers=2)
        loader_teste = DataLoader(teste, batch_size=4, shuffle=False, num_workers=2)

        return loader_treino, loader_teste
