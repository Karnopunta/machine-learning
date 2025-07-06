import torch

# Load model dengan mematikan mode 'weights_only' (izin penuh)
model = torch.load('best.pt', map_location='cpu', weights_only=False)

# Simpan ulang model (untuk kompatibilitas lintas platform)
torch.save(model, 'best_windows.pt')

print("Model berhasil dikonversi dan disimpan sebagai best_windows.pt")
