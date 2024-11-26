from setuptools import setup, find_packages

setup(
    name="fearless_ep",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "numpyro",
        "plum-dispatch",
        "expfam @ git+https://github.com/jonny-so/expfam-jax.git#egg=expfam",
        "jaxutil @ git+https://github.com/jonny-so/jaxutil.git#egg=jaxutil",
    ],
)
