""" Models for Siamese Network
    - Embedding: Feature extractor and vectorizer
    - L1Dist: Computes the absolute difference between two embeddings
    - Siamese: Combines the embedding and distance modules, and includes a classifier
"""
import torch
from torch import nn

class Embedding(nn.Module):
    """ Feature extractor for the Siamese Network
        - Convolutional layers with ReLU activations and max pooling
        - Fully connected layer to produce the embedding vector
    """
    def __init__(self, in_channels=3, embedding_dim=4096):
        """ Initialize the Embedding module
            Args:
                in_channels (int): Number of input channels (default: 3 for RGB images)
                embedding_dim (int): Dimension of the output embedding vector (default: 4096)
        """
        super().__init__()

        self.Feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=10),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2), 

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            nn.ReLU()  
        )

        self.Feature_vector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=9216, out_features=embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Forward pass through the embedding module
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            Returns:
                torch.Tensor: Output embedding vector of shape (batch_size, embedding_dim)
        """
        x = self.Feature_extractor(x)
        x = self.Feature_vector(x)
        return x


class L1Dist(nn.Module):
    """ Computes the absolute difference between two embeddings
        - Used to measure the similarity between two input images
    """
    def __init__(self):
        """Initialize the L1Dist module"""
        super().__init__()
    
    def forward(self, in_embedding, val_embedding):
        """ Forward pass to compute the absolute difference between two embeddings
            Args:
                in_embedding (torch.Tensor): First embedding vector of shape (batch_size, embedding_dim)
                val_embedding (torch.Tensor): Second embedding vector of shape (batch_size, embedding_dim)
            Returns:
                torch.Tensor: Absolute difference between the two embeddings
        """
        return torch.abs(in_embedding - val_embedding)    # element-wise absolute difference


class Siamese(nn.Module):
    """ Combines the embedding and distance modules, and includes a classifier
        - Takes two input images, computes their embeddings, and classifies the similarity score
    """
    def __init__(self, in_channels=3, embedding_dim=4096):
        """ Initialize the Siamese module
            Args:
                in_channels (int): Number of input channels (default: 3 for RGB images)
                embedding_dim (int): Dimension of the output embedding vector (default: 4096)
        """
        super().__init__()
        self.Embedding = Embedding(in_channels=in_channels, embedding_dim=embedding_dim)
        self.L1Dist = L1Dist()
        self.Classifier = nn.Linear(in_features=4096, out_features=1)

    def forward(self, x1, x2):
        """ Forward pass through the Siamese module
            Args:
                x1 (torch.Tensor): First input tensor of shape (batch_size, in_channels, height, width)
                x2 (torch.Tensor): Second input tensor of shape (batch_size, in_channels, height, width)
            Returns:
                torch.Tensor: Similarity score between the two input images
        """
        x1_embedding = self.Embedding(x1)
        x2_embedding = self.Embedding(x2)

        l1_dis = self.L1Dist(x1_embedding, x2_embedding)

        score  = self.Classifier(l1_dis) 
        return score 