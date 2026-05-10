{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNbp7rreMVWjZUOlt6p+mPN",
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
        "<a href=\"https://colab.research.google.com/github/martinezdiemar-afk/mi-ia-web/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 400
        },
        "id": "glbkxWPp8XB_",
        "outputId": "c283d6dd-4dab-4146-dc08-e5520dd967a7"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "No module named 'streamlit'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_15015/1695765030.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mstreamlit\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mst\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtensorflow\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mstreamlit_drawable_canvas\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mst_canvas\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mcv2\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mnumpy\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ],
      "source": [
        "import streamlit as st\n",
        "import tensorflow as tf\n",
        "from streamlit_drawable_canvas import st_canvas\n",
        "import cv2\n",
        "import numpy as np\n",
        "\n",
        "# Configuración visual\n",
        "st.set_page_config(page_title=\"IA Reconocedor de Dígitos\")\n",
        "st.title(\"🖌️ Reconocedor de Dígitos en Tiempo Real\")\n",
        "st.write(\"Dibuja un número en el cuadro negro y la IA de la Práctica 11 lo adivinará.\")\n",
        "\n",
        "# 1. Cargar el modelo que guardaste\n",
        "@st.cache_resource\n",
        "def load_my_model():\n",
        "    # El archivo debe llamarse exactamente así en GitHub\n",
        "    return tf.keras.models.load_model('modelo_mnist.keras')\n",
        "\n",
        "try:\n",
        "    model = load_my_model()\n",
        "except Exception as e:\n",
        "    st.error(\"No se encontró el archivo 'modelo_mnist.keras'. Asegúrate de subirlo a GitHub.\")\n",
        "    st.stop()\n",
        "\n",
        "# 2. Configurar el lienzo de dibujo\n",
        "st.subheader(\"Dibuja aquí:\")\n",
        "canvas_result = st_canvas(\n",
        "    fill_color=\"white\",\n",
        "    stroke_width=20,\n",
        "    stroke_color=\"white\",\n",
        "    background_color=\"black\",\n",
        "    height=280,\n",
        "    width=280,\n",
        "    drawing_mode=\"freedraw\",\n",
        "    key=\"canvas\",\n",
        ")\n",
        "\n",
        "# 3. Procesar la imagen cuando el usuario dibuja\n",
        "if canvas_result.image_data is not None:\n",
        "    # Obtener los datos del dibujo\n",
        "    img_raw = canvas_result.image_data.astype('uint8')\n",
        "\n",
        "    # Redimensionar a 28x28 (como el dataset MNIST)\n",
        "    img_resizing = cv2.resize(img_raw, (28, 28))\n",
        "\n",
        "    # Pasar a escala de grises\n",
        "    img_gray = cv2.cvtColor(img_resizing, cv2.COLOR_BGR2GRAY)\n",
        "\n",
        "    # Normalizar (valores entre 0 y 1)\n",
        "    img_normalized = img_gray / 255.0\n",
        "\n",
        "    # Preparar para el modelo (añadir dimensiones de batch y canal)\n",
        "    img_input = img_normalized.reshape(1, 28, 28, 1)\n",
        "\n",
        "    if st.button('Predecir número'):\n",
        "        # Realizar la predicción\n",
        "        prediction = model.predict(img_input)\n",
        "        clase_predicha = np.argmax(prediction)\n",
        "        probabilidad = np.max(prediction)\n",
        "\n",
        "        # 4. Mostrar resultados según la confianza\n",
        "        st.divider()\n",
        "        if probabilidad > 0.8:\n",
        "            st.success(f\"### ¡Es un {clase_predicha}! (Confianza: {probabilidad:.2%})\")\n",
        "        elif probabilidad > 0.4:\n",
        "            st.warning(f\"### Parece un {clase_predicha}, pero no estoy muy segura ({probabilidad:.2%})\")\n",
        "        else:\n",
        "            st.error(f\"### No estoy segura. ¿Es un {clase_predicha}? (Confianza muy baja)\")\n",
        "\n",
        "        # Mostrar gráfico de barras con todas las probabilidades\n",
        "        st.bar_chart(prediction[0])"
      ]
    }
  ]
}