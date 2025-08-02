import torch


class StandardizeTransform:
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """
        Standardizes the input data y.
        
        Args:
            y (torch.Tensor): Input data to be standardized.
            
        Returns:
            torch.Tensor: Standardized data.
        """
        mean = y.mean()
        std = y.std()
        return (y - mean) / std


class LogTransform:
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """
        Applies a logarithmic transformation to the input data y.
        
        Args:
            y (torch.Tensor): Input data to be transformed.
            
        Returns:
            torch.Tensor: Log-transformed data.
        """
        if (y <= 0).any():
            print("Warning: Log transformation is not defined for non-positive values. Values will be shifted.")
            # Shift non-positive values to positive values
            y = y - y.min() + 1e-6

        return torch.log1p(y)


class BilogTransform:
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """
        Applies a bilogarithmic transformation to the input data y.
        
        Args:
            y (torch.Tensor): Input data to be transformed.
            
        Returns:
            torch.Tensor: Bilogarithmically transformed data.
        """
        return y.sign() * torch.log1p(y.abs())