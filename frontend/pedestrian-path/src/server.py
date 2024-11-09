from flask import Flask, request, send_file
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route('/')
def index():
    return 'Flask server is running'

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Возвращаем пустой ответ с кодом 204 (No Content)

@app.route('/generate_map', methods=['POST'])
def generate_map():
    data = request.json
    coordinates = data['coordinates']
    color = data['color']

    # Генерация графика с использованием matplotlib
    fig, ax = plt.subplots()

    # Разделим координаты на X и Y
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    ax.scatter(x_coords, y_coords, color=color)

    # Преобразование графика в изображение
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
