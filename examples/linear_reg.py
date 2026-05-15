import time

import mlx.core as mx

time_start = time.perf_counter()
num_features = 100
num_examples = 1_000
num_iters = 10_000
learning_rate = 0.01

w_star = mx.random.normal((num_features,))

X = mx.random.normal((num_examples, num_features))

eps = 1e-2 * mx.random.normal((num_examples,))
y = X @ w_star + eps

w = 1e-2 * mx.random.normal((num_features,))

def loss_fn(w):
    return 0.5 * mx.mean(mx.square(X @ w - y))

grad_fn = mx.grad(loss_fn)

tic = time.perf_counter()
for _ in range(num_iters):
    grad = grad_fn(w)
    w = w - learning_rate * grad
    mx.eval(w)
toc = time.perf_counter()

loss = loss_fn(w)
error_norm = mx.sum(mx.square(w - w_star), stream=mx.gpu).item() ** 0.5
throughput = num_iters / (toc - tic)

print(
    f"Loss {loss.item():.5f}, L2 distance: |w-w*| = {error_norm:.5f}, "
    f"Throughput {throughput:.5f} (it/s)"
)
time_end = time.perf_counter()
print(f"Total time: {time_end - time_start:.5f} seconds")
