{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPLdSXr2KI7VBbOQ+Xs6YVE",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/AlfredoV3g4/ExamenFinal/blob/main/examen.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Examen Final\n"
      ],
      "metadata": {
        "id": "QIvC0JZ-zHH-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "import numpy as np"
      ],
      "metadata": {
        "id": "SJ3y_di0zNqx"
      },
      "execution_count": 42,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "centimetros = np.array([2.54, 38.1, 69.85, 127, 12.7, 25.4,1], dtype=float)\n",
        "pulgadas = np.array([1, 15, 27.5, 50, 5,10,0.393701], dtype=float)"
      ],
      "metadata": {
        "id": "NQNkJwMI0LXr"
      },
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "capa = tf.keras.layers.Dense(units=1, input_shape=[1])\n",
        "modelo = tf.keras.Sequential([capa])"
      ],
      "metadata": {
        "id": "Z6ympslp1DUe"
      },
      "execution_count": 44,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "modelo.compile(\n",
        "    optimizer = tf.keras.optimizers.Adam(0.1),\n",
        "    loss='mean_squared_error'\n",
        ")"
      ],
      "metadata": {
        "id": "phN_7pvt1I2H"
      },
      "execution_count": 45,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"comenzando entrenamiento\")\n",
        "historial=modelo.fit(centimetros, pulgadas, epochs=1000, verbose=False)\n",
        "print(\"modelo entrenado!!!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pYexSAZt1U1_",
        "outputId": "a89e11bb-95ea-4d5c-9849-14e16b24be30"
      },
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "comenzando entrenamiento\n",
            "modelo entrenado!!!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "plt.xlabel(\"# Epoca\")\n",
        "plt.ylabel(\"Magnitud de perdida\")\n",
        "plt.plot(historial.history[\"loss\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "6iFsuliV1lLn",
        "outputId": "46fe0c18-58c1-41a9-8bcf-ed584e51f504"
      },
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x79b8b0dd5fc0>]"
            ]
          },
          "metadata": {},
          "execution_count": 47
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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7GUGAIB5RgPRwYMH1bFjR82cOfOUbfr27auSkhL79eKLL/qcHzJkiLZu3ar8/HwtXrxYK1as0B133GGfd7vd6tOnjzIzM7Vu3TpNnz5dkyZN0qxZs+rse/mDvcwAADAv3OSH9+vXT/369fvRNhEREUpJSTnpuY8++khvvvmmPvzwQ3Xt2lWS9NRTT+mqq67S448/rrS0NM2bN0+VlZWaPXu2XC6X2rZtq8LCQs2YMcMnOB2voqJCFRUV9nu3232G3/D02MsMAADzgn4M0bvvvqukpCRlZ2frrrvu0p49e+xzBQUFio+Pt8OQJPXu3VtOp1OrV6+22/To0UMul8tuk5ubq6KiIn333Xcn/cy8vDzFxcXZr/T09Dr6dsdt3UEiAgDAmKAORH379tU///lPLV26VH/84x+1fPly9evXTzU1NZKk0tJSJSUl+fxOeHi4GjdurNLSUrtNcnKyTxvve2+bHxo/frzKy8vt165duwL91WwOVqoGAMA4o4/MTufGG2+0f27fvr06dOigli1b6t1331WvXr3q7HMjIiIUERFRZ9c/ntNeqZpEBACAKUHdQ/RDLVq0UGJioj799FNJUkpKinbv3u3Tprq6Wnv37rXHHaWkpKisrMynjff9qcYm1SfvtHviEAAA5pxVgejLL7/Unj17lJqaKknKycnRvn37tG7dOrvNsmXL5PF41K1bN7vNihUrVFVVZbfJz89Xdna2EhIS6vcLnAR7mQEAYJ7RQHTgwAEVFhaqsLBQkrRz504VFhaquLhYBw4c0P33369Vq1bp888/19KlSzVgwACdf/75ys3NlSS1bt1affv21e233641a9bogw8+0KhRo3TjjTcqLS1NkjR48GC5XC6NGDFCW7du1UsvvaQnn3xSY8eONfW1fTjYywwAAOOMBqK1a9eqc+fO6ty5syRp7Nix6ty5syZOnKiwsDBt2rRJ11xzjS688EKNGDFCXbp00XvvveczvmfevHlq1aqVevXqpauuukqXXXaZzxpDcXFxevvtt7Vz50516dJF9913nyZOnHjKKff1zZ52b7QKAABCm8NiNO9pud1uxcXFqby8XLGxsQG99i+fek9bvnJrzi0Xq2d20ul/AQAA1Io/f7/PqjFE5yLvoGq6iAAAMIdAZJiThRkBADCOQGQaCzMCAGAcgcgwBlUDAGAegcgwHpkBAGAe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z5+v9evX68MPPzzhHPc5cHbs2KFnnnlGY8eO1e9+9zt9+OGHuueee+RyuTR8+HD7Xp3sXh5/r5OSknzOh4eHq3Hjxtzrox588EG53W61atVKYWFhqqmp0aOPPqohQ4ZIEve5DgTqnpaWliorK+uEa3jPJSQknHGNBCKcE0aOHKktW7bo/fffN13KOWfXrl269957lZ+fr8jISNPlnNM8Ho+6du2qxx57TJLUuXNnbdmyRc8++6yGDx9uuLpzx7///W/NmzdPL7zwgtq2bavCwkKNHj1aaWlp3OcQxiMzAxITExUWFnbCLJyysjKlpKQYqursNWrUKC1evFjvvPOOmjVrZh9PSUlRZWWl9u3b59P++PuckpJy0n8H7zkceSS2e/duXXTRRQoPD1d4eLiWL1+uv/zlLwoPD1dycjL3OUBSU1PVpk0bn2OtW7dWcXGxpGP36sf+25GSkqLdu3f7nK+urtbevXu510fdf//9evDBB3XjjTeqffv2uummmzRmzBjl5eVJ4j7XhUDd07r8bwmByACXy6UuXbpo6dKl9jGPx6OlS5cqJyfHYGVnF8uyNGrUKL388statmzZCd2oXbp0UYMGDXzuc1FRkYqLi+37nJOTo82bN/v8nzA/P1+xsbEn/GEKVb169dLmzZtVWFhov7p27aohQ4bYP3OfA+PnP//5CUtHfPzxx8rMzJQkZWVlKSUlxedeu91urV692ude79u3T+vWrbPbLFu2TB6PR926dauHbxH8Dh06JKfT989fWFiYPB6PJO5zXQjUPc3JydGKFStUVVVlt8nPz1d2dvZPelwmiWn3psyfP9+KiIiw5s6da23bts264447rPj4eJ9ZOPhxd911lxUXF2e9++67VklJif06dOiQ3ebOO++0MjIyrGXLlllr1661cnJyrJycHPu8dzp4nz59rMLCQuvNN9+0mjZtynTw0zh+lpllcZ8DZc2aNVZ4eLj16KOPWp988ok1b948Kyoqynr++eftNlOnTrXi4+OtV155xdq0aZM1YMCAk05d7ty5s7V69Wrr/fffty644IKQng7+Q8OHD7fOO+88e9r9woULrcTEROuBBx6w23Cf/bd//35rw4YN1oYNGyxJ1owZM6wNGzZYX3zxhWVZgbmn+/bts5KTk62bbrrJ2rJlizV//nwrKiqKafdnu6eeesrKyMiwXC6Xdckll1irVq0yXdJZRdJJX3PmzLHbfP/999bdd99tJSQkWFFRUdZ1111nlZSU+Fzn888/t/r162c1bNjQSkxMtO677z6rqqqqnr/N2eWHgYj7HDivvfaa1a5dOysiIsJq1aqVNWvWLJ/zHo/HmjBhgpWcnGxFRERYvXr1soqKinza7Nmzxxo0aJDVqFEjKzY21rrlllus/fv31+fXCGput9u69957rYyMDCsyMtJq0aKF9fvf/95nKjf32X/vvPPOSf+bPHz4cMuyAndPN27caF122WVWRESEdd5551lTp04NSP0OyzpuaU4AAIAQxBgiAAAQ8ghEAAAg5BGIAABAyCMQAQCAkEcgAgAAIY9ABAAAQh6BCAAAhDwCEQAACHkEIgAAEPIIRACC2jfffCOXy6WDBw+qqqpK0dHR9u7vpzJp0iQ5HI4TXq1ataqnqgGcbcJNFwAAP6agoEAdO3ZUdHS0Vq9ercaNGysjI+O0v9e2bVstWbLE51h4OP/JA3By9BABCGorV67Uz3/+c0nS+++/b/98OuHh4UpJSfF5JSYm2uebN2+uKVOmaNCgQYqOjtZ5552nmTNn+lyjuLhYAwYMUKNGjRQbG6tf/epXKisr82nz2muv6eKLL1ZkZKQSExN13XXX2ef+9a9/qWvXroqJiVFKSooGDx6s3bt3n+mtAFCHCEQAgk5xcbHi4+MVHx+vGTNm6G9/+5vi4+P1u9/9TosWLVJ8fLzuvvvun/w506dPV8eOHbVhwwY9+OCDuvfee5Wfny9J8ng8GjBggPbu3avly5crPz9fO3bs0K9//Wv7919//XVdd911uuqqq7RhwwYtXbpUl1xyiX2+qqpKU6ZM0caNG7Vo0SJ9/vnnuvnmm39y3QACj93uAQSd6upqffnll3K73eratavWrl2r6OhoderUSa+//royMjLUqFEjnx6f402aNElTpkxRw4YNfY4PHTpUzz77rKQjPUStW7fWf//7X/v8jTfeKLfbrTfeeEP5+fnq16+fdu7cqfT0dEnStm3b1LZtW61Zs0YXX3yxLr30UrVo0ULPP/98rb7X2rVrdfHFF2v//v1q1KjRmdwaAHWEHiIAQSc8PFzNmzfX9u3bdfHFF6tDhw4qLS1VcnKyevTooebNm58yDHllZ2ersLDQ5/XII4/4tMnJyTnh/UcffSRJ+uijj5Senm6HIUlq06aN4uPj7TaFhYXq1avXKWtYt26drr76amVkZCgmJkaXX365JJ12UDiA+scIQwBBp23btvriiy9UVVUlj8ejRo0aqbq6WtXV1WrUqJEyMzO1devWH72Gy+XS+eefX6d1/rAH6ngHDx5Ubm6ucnNzNW/ePDVt2lTFxcXKzc1VZWVlndYFwH/0EAEIOm+88YYKCwuVkpKi559/XoWFhWrXrp2eeOIJFRYW6o033gjI56xateqE961bt5YktW7dWrt27dKuXbvs89u2bdO+ffvUpk0bSVKHDh20dOnSk157+/bt2rNnj6ZOnaru3burVatWDKgGghg9RACCTmZmpkpLS1VWVqYBAwbI4XBo69atGjhwoFJTU2t1jerqapWWlvocczgcSk5Ott9/8MEHmjZtmq699lrl5+drwYIFev311yVJvXv3Vvv27TVkyBA98cQTqq6u1t13363LL79cXbt2lSQ9/PDD6tWrl1q2bKkbb7xR1dXVeuONNzRu3DhlZGTI5XLpqaee0p133qktW7ZoypQpAbpDAAKNHiIAQendd9+1p7OvWbNGzZo1q3UYkqStW7cqNTXV55WZmenT5r777tPatWvVuXNn/eEPf9CMGTOUm5sr6Uh4euWVV5SQkKAePXqod+/eatGihV566SX796+44gotWLBAr776qjp16qRf/OIXWrNmjSSpadOmmjt3rhYsWKA2bdpo6tSpevzxxwNwZwDUBWaZAQhJzZs31+jRozV69GjTpQAIAvQQAQCAkEcgAgAAIY9HZgAAIOTRQwQAAEIegQgAAIQ8AhEAAAh5BCIAABDyCEQAACDkEYgAAEDIIxABAICQRyACAAAh7/8DZpn6rUCjgpoAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"realizar una prediccion!!!\")\n",
        "resultado = modelo.predict([100.0])\n",
        "print (\"el resultado es\" + str(resultado) + \"Pesos Colombianas!!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6C1iRzNV1sNV",
        "outputId": "00326675-a1ae-4d67-c85a-608db2461170"
      },
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "realizar una prediccion!!!\n",
            "1/1 [==============================] - 0s 36ms/step\n",
            "el resultado es[[39.37008]]Pesos Colombianas!!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "modelo.save('centimetros_a_pulgadas.h5')"
      ],
      "metadata": {
        "id": "qyWpz81D14Im"
      },
      "execution_count": 51,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!ls"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "U7-I330p4jId",
        "outputId": "fd63ce83-e576-4f4d-8cfc-2b5a87d4ab91"
      },
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "centimetros_a_pulgadas.h5  sample_data\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install tensorflowjs"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "q22PXnMt4m4m",
        "outputId": "927b4662-6879-4b7e-ba0b-2be4c7ba6234"
      },
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: tensorflowjs in /usr/local/lib/python3.10/dist-packages (4.14.0)\n",
            "Requirement already satisfied: flax>=0.7.2 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (0.7.5)\n",
            "Requirement already satisfied: importlib_resources>=5.9.0 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (6.1.1)\n",
            "Requirement already satisfied: jax>=0.4.13 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (0.4.20)\n",
            "Requirement already satisfied: jaxlib>=0.4.13 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (0.4.20+cuda11.cudnn86)\n",
            "Requirement already satisfied: tensorflow<3,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (2.15.0)\n",
            "Requirement already satisfied: tensorflow-decision-forests>=1.5.0 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (1.8.1)\n",
            "Requirement already satisfied: six<2,>=1.16.0 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (1.16.0)\n",
            "Requirement already satisfied: tensorflow-hub>=0.14.0 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (0.15.0)\n",
            "Requirement already satisfied: packaging~=23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflowjs) (23.2)\n",
            "Requirement already satisfied: numpy>=1.22 in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (1.23.5)\n",
            "Requirement already satisfied: msgpack in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (1.0.7)\n",
            "Requirement already satisfied: optax in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (0.1.7)\n",
            "Requirement already satisfied: orbax-checkpoint in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (0.4.3)\n",
            "Requirement already satisfied: tensorstore in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (0.1.45)\n",
            "Requirement already satisfied: rich>=11.1 in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (13.7.0)\n",
            "Requirement already satisfied: typing-extensions>=4.2 in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (4.5.0)\n",
            "Requirement already satisfied: PyYAML>=5.4.1 in /usr/local/lib/python3.10/dist-packages (from flax>=0.7.2->tensorflowjs) (6.0.1)\n",
            "Requirement already satisfied: ml-dtypes>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from jax>=0.4.13->tensorflowjs) (0.2.0)\n",
            "Requirement already satisfied: opt-einsum in /usr/local/lib/python3.10/dist-packages (from jax>=0.4.13->tensorflowjs) (3.3.0)\n",
            "Requirement already satisfied: scipy>=1.9 in /usr/local/lib/python3.10/dist-packages (from jax>=0.4.13->tensorflowjs) (1.11.4)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (1.4.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (1.6.3)\n",
            "Requirement already satisfied: flatbuffers>=23.5.26 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (23.5.26)\n",
            "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (0.5.4)\n",
            "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (0.2.0)\n",
            "Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (3.9.0)\n",
            "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (16.0.6)\n",
            "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (3.20.3)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (67.7.2)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (2.3.0)\n",
            "Requirement already satisfied: wrapt<1.15,>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (1.14.1)\n",
            "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (0.34.0)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (1.59.3)\n",
            "Requirement already satisfied: tensorboard<2.16,>=2.15 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (2.15.1)\n",
            "Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (2.15.0)\n",
            "Requirement already satisfied: keras<2.16,>=2.15.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow<3,>=2.13.0->tensorflowjs) (2.15.0)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from tensorflow-decision-forests>=1.5.0->tensorflowjs) (1.5.3)\n",
            "Requirement already satisfied: wheel in /usr/local/lib/python3.10/dist-packages (from tensorflow-decision-forests>=1.5.0->tensorflowjs) (0.42.0)\n",
            "Requirement already satisfied: wurlitzer in /usr/local/lib/python3.10/dist-packages (from tensorflow-decision-forests>=1.5.0->tensorflowjs) (3.0.3)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=11.1->flax>=0.7.2->tensorflowjs) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=11.1->flax>=0.7.2->tensorflowjs) (2.16.1)\n",
            "Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (2.17.3)\n",
            "Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (1.0.0)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (3.5.1)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (2.31.0)\n",
            "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (0.7.2)\n",
            "Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (3.0.1)\n",
            "Requirement already satisfied: chex>=0.1.5 in /usr/local/lib/python3.10/dist-packages (from optax->flax>=0.7.2->tensorflowjs) (0.1.7)\n",
            "Requirement already satisfied: etils[epath,epy] in /usr/local/lib/python3.10/dist-packages (from orbax-checkpoint->flax>=0.7.2->tensorflowjs) (1.5.2)\n",
            "Requirement already satisfied: nest_asyncio in /usr/local/lib/python3.10/dist-packages (from orbax-checkpoint->flax>=0.7.2->tensorflowjs) (1.5.8)\n",
            "Requirement already satisfied: python-dateutil>=2.8.1 in /usr/local/lib/python3.10/dist-packages (from pandas->tensorflow-decision-forests>=1.5.0->tensorflowjs) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->tensorflow-decision-forests>=1.5.0->tensorflowjs) (2023.3.post1)\n",
            "Requirement already satisfied: dm-tree>=0.1.5 in /usr/local/lib/python3.10/dist-packages (from chex>=0.1.5->optax->flax>=0.7.2->tensorflowjs) (0.1.8)\n",
            "Requirement already satisfied: toolz>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from chex>=0.1.5->optax->flax>=0.7.2->tensorflowjs) (0.12.0)\n",
            "Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (5.3.2)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (0.3.0)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (4.9)\n",
            "Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (1.3.1)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=11.1->flax>=0.7.2->tensorflowjs) (0.1.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (2023.11.17)\n",
            "Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (2.1.3)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from etils[epath,epy]->orbax-checkpoint->flax>=0.7.2->tensorflowjs) (2023.6.0)\n",
            "Requirement already satisfied: zipp in /usr/local/lib/python3.10/dist-packages (from etils[epath,epy]->orbax-checkpoint->flax>=0.7.2->tensorflowjs) (3.17.0)\n",
            "Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (0.5.1)\n",
            "Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.10/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow<3,>=2.13.0->tensorflowjs) (3.2.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!mkdir mediciones"
      ],
      "metadata": {
        "id": "QfiKURmT4-K2"
      },
      "execution_count": 54,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!tensorflowjs_converter --input_format keras centimetros_a_pulgadas.h5 mediciones"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "M4hw7nzY5Dde",
        "outputId": "692fe42f-abdc-4210-995f-f34abae30802"
      },
      "execution_count": 55,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2023-12-04 22:55:20.910754: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
            "2023-12-04 22:55:20.910847: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
            "2023-12-04 22:55:20.912831: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
            "2023-12-04 22:55:24.010139: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!ls mediciones"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Y3yP7-9b5PTg",
        "outputId": "6b516183-3bb1-4ab8-dc85-cf4fc090532a"
      },
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "group1-shard1of1.bin  model.json\n"
          ]
        }
      ]
    }
  ]
}