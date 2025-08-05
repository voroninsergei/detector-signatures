from flask import Flask, render_template_string, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from modules import basic_analysis

# Папка для загрузки временных изображений
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Создаем папку для загрузок, если её нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Проверяет, соответствует ли файл допустимым расширениям."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Главная страница позволяет загрузить изображение рукописного текста или подписи.
    После загрузки изображение сохраняется на сервере, и запускается базовый анализ.
    На этапе 1 выполняется только сохранение и возврат тестового результата.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Вызываем базовый анализ
            result = basic_analysis.analyze_handwriting(filepath)
            return render_template_string(
                """
                <h1>Результаты анализа</h1>
                <p>{{ result }}</p>
                <a href="{{ url_for('index') }}">Загрузить другое изображение</a>
                """,
                result=result
            )
    return render_template_string(
        """
        <h1>Детектор подписей</h1>
        <p>Загрузите изображение рукописного текста или подписи для анализа.</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <input type="submit" value="Анализировать">
        </form>
        """
    )


if __name__ == '__main__':
    # Запуск приложения в режиме отладки
    app.run(debug=True)