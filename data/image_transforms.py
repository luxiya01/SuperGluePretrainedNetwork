import torch


class ColumnwiseNormalization:
    """Normalize input image by dividing each column with the column mean and clipping
    the resulting values between (0, a_max)."""

    def __init__(self, a_max):
        self.a_max = a_max

    def __call__(self, image):
        # image shape = (1, H (num_pings), W (num_bins))
        col_mean = image.mean(axis=1)
        image = torch.nan_to_num(image / col_mean)
        return torch.clip(image, min=0., max=self.a_max)
