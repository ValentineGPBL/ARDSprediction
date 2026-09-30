import torch


class ModelUtils:
    @staticmethod
    def load_all_images(df, cnn_model):
        tensors = torch.stack([cnn_model.load_image(path) for path in df['image_path']])
        return tensors