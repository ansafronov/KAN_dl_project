# LAN_dl_project


LAN (Learnable activation networks) are almost MLP’s but
with learnable activation functions parametrized as splines.
About activation functions:
1. Each activation function has the form:
                    $$\phi(x) = SiLU(x) + spline(x)$$
2. $spline(x)$ is linear combination of b-splines of degree $3$:
                    $$spline(x) = \sum_{i} c_i B_i(x)$$
3. Coefficients $c_i$ are learnable, the number of splines is a parameter for each activation function.

Size of LAN:

For a LAN with width $N$, depth $L$, and grid point number $G$, the number of parameters is $N^2L + NLG$ where $N^2L$ is the number of parameters for weight matrices and $NLG$ is the number of parameters for spline activations.

About realisation:

You can see our implementation of a LAN Layer (LAN activation function) in `LAN_layer.ipynb`. For splines, we use `KAN_spline.py` from `pykan` library. Usage is simple: after Linear layer, you Use LAN-Layer TODO


