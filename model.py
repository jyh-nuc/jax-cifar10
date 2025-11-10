import flax.linen as nn

class OptimizedCNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, deterministic=False):
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=0.25, deterministic=deterministic)(x)
        
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=0.25, deterministic=deterministic)(x)
        
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=deterministic)(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x
