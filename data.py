import jax
import jax.numpy as jnp

def generate_synthetic_data(train=True):
    rng = jax.random.PRNGKey(0)
    if train:
        x = jax.random.uniform(rng, (60000, 32, 32, 3))
        y = jax.random.randint(rng, (60000,), 0, 10)
    else:
        x = jax.random.uniform(rng, (10000, 32, 32, 3))
        y = jax.random.randint(rng, (10000,), 0, 10)
    return (x, y)

def create_data_loader(data, batch_size):
    x, y = data
    num_batches = len(x) // batch_size
    for i in range(num_batches):
        yield (x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])

def evaluate(state, data_loader, deterministic=True):
    correct = 0
    total = 0
    for batch in data_loader:
        x, y = batch
        logits = state.apply_fn({'params': state.params}, x, deterministic=deterministic)
        predictions = jnp.argmax(logits, axis=1)
        correct += jnp.sum(predictions == y)
        total += len(y)
    return correct / total
