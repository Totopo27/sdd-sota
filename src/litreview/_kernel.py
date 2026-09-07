"""Instalación del kernel de IPython para Jupyter."""

import sys
from ipykernel.kernelspec import install


def install_kernel() -> None:
    """Instala el kernel Jupyter en ~/.local/share/jupyter/kernels/shared_env."""
    print("Instalando kernel Jupyter en ~/.local/share/jupyter/kernels/litreview_env ...")
    install(
        user=True,
        kernel_name="litreview_env",
        display_name="Python (litreview_env)",
    )
    print("Kernel instalado exitosamente.")


if __name__ == "__main__":
    install_kernel()
    