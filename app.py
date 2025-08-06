from flask import Flask, render_template_string, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import base64
from io import BytesIO
from typing import Tuple

import cv2
from modules import basic_analysis, comparative, preprocessing

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


def _image_to_base64(img) -> str:
    """Преобразует изображение NumPy в строку base64 для вставки в HTML."""
    if img is None:
        return ''
    # Кодируем в PNG в буфер
    ret, buffer = cv2.imencode('.png', img)
    if not ret:
        return ''
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64}"


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
            # Дополнительно генерируем изображения: исходное, бинаризованное, сегментация
            try:
                original_img = preprocessing.load_image(filepath)
                processed_img = preprocessing.preprocess_image(original_img)
                # Подготавливаем изображение сегментации: копия исходного + прямоугольники вокруг строк
                seg_overlay = original_img.copy()
                lines = preprocessing.segment_text(processed_img)
                y_offset = 0
                # Для каждого сегмента найдём его вертикальные координаты относительно общего изображения
                # segment_text возвращает список отдельных изображений строк, но без координат.
                # Для простоты вычислим горизонтальную проекцию снова.
                proj = (processed_img // 255).sum(axis=1)
                max_val = proj.max() if proj.size > 0 else 0
                threshold = max_val * 0.1 if max_val else 0
                in_line = False
                start = 0
                for idx, val in enumerate(proj):
                    if val > threshold and not in_line:
                        in_line = True
                        start = idx
                    elif val <= threshold and in_line:
                        end = idx
                        cv2.rectangle(seg_overlay, (0, start), (seg_overlay.shape[1]-1, end), (0, 255, 0), 2)
                        in_line = False
                if in_line:
                    end = len(proj)-1
                    cv2.rectangle(seg_overlay, (0, start), (seg_overlay.shape[1]-1, end), (0, 255, 0), 2)
                # Конвертируем изображения в base64
                original_b64 = _image_to_base64(original_img)
                processed_b64 = _image_to_base64(cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR))
                overlay_b64 = _image_to_base64(seg_overlay)
            except Exception:
                original_b64 = processed_b64 = overlay_b64 = ''
            return render_template_string(
                """
                <h1>Результаты анализа</h1>
                <p>{{ result.replace('\n', '<br>')|safe }}</p>
                {% if original_b64 %}
                <h2>Исходное изображение</h2>
                <img src="{{ original_b64 }}" style="max-width:45%; height:auto; border:1px solid #ccc;">
                {% endif %}
                {% if processed_b64 %}
                <h2>Бинаризованное изображение</h2>
                <img src="{{ processed_b64 }}" style="max-width:45%; height:auto; border:1px solid #ccc;">
                {% endif %}
                {% if overlay_b64 %}
                <h2>Сегментация строк</h2>
                <img src="{{ overlay_b64 }}" style="max-width:45%; height:auto; border:1px solid #ccc;">
                {% endif %}
                <p><a href="{{ url_for('index') }}">Загрузить другое изображение</a> | <a href="{{ url_for('compare') }}">Сравнение</a></p>
                """,
                result=result,
                original_b64=original_b64,
                processed_b64=processed_b64,
                overlay_b64=overlay_b64
            )
    return render_template_string(
        """
        <h1>Детектор подписей</h1>
        <p>Загрузите изображение рукописного текста или подписи для анализа.</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <input type="submit" value="Анализировать">
        </form>
        <p>Или <a href="{{ url_for('compare') }}">перейдите к сравнительному анализу двух изображений</a>.</p>
        """
    )


if __name__ == '__main__':
    # Запуск приложения в режиме отладки
    app.run(debug=True)


# Новый маршрут для сравнения двух изображений
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """
    Форма для загрузки двух изображений (спорного и образца) и выполнения
    сравнительного анализа. Результат отображается в виде коэффициента
    сходства и его словесной интерпретации.
    """
    if request.method == 'POST':
        file_a = request.files.get('file_a')
        file_b = request.files.get('file_b')
        if not file_a or not file_b:
            return redirect(request.url)
        if file_a.filename == '' or file_b.filename == '':
            return redirect(request.url)
        if allowed_file(file_a.filename) and allowed_file(file_b.filename):
            filename_a = secure_filename(file_a.filename)
            filename_b = secure_filename(file_b.filename)
            path_a = os.path.join(app.config['UPLOAD_FOLDER'], filename_a)
            path_b = os.path.join(app.config['UPLOAD_FOLDER'], filename_b)
            file_a.save(path_a)
            file_b.save(path_b)
            try:
                similarity, details = comparative.compare_images(path_a, path_b)
                interpretation = comparative.interpret_similarity(similarity)
                # Подготовим изображения для вывода
                try:
                    img_a = preprocessing.load_image(path_a)
                    img_b = preprocessing.load_image(path_b)
                    img_a_b64 = _image_to_base64(img_a)
                    img_b_b64 = _image_to_base64(img_b)
                except Exception:
                    img_a_b64 = img_b_b64 = ''
                return render_template_string(
                    """
                    <h1>Сравнительный анализ</h1>
                    <p>Коэффициент сходства: {{ similarity:.2f }}</p>
                    <p>Интерпретация: {{ interpretation }}</p>
                    <h2>Исходные изображения</h2>
                    {% if img_a_b64 %}<img src="{{ img_a_b64 }}" style="max-width:45%; border:1px solid #ccc;">{% endif %}
                    {% if img_b_b64 %}<img src="{{ img_b_b64 }}" style="max-width:45%; border:1px solid #ccc;">{% endif %}
                    <h2>Детали</h2>
                    <ul>
                        <li>Средний размер букв: {{ details['size_avg'][0]:.1f }} vs {{ details['size_avg'][1]:.1f }}</li>
                        <li>Разгон между буквами: {{ details['spacing_avg'][0]:.1f }} vs {{ details['spacing_avg'][1]:.1f }}</li>
                        <li>Наклон: {{ details['slant'][0]:.1f }}° vs {{ details['slant'][1]:.1f }}°</li>
                        <li>Коэффициент связности: {{ details['connectivity'][0]:.2f }} vs {{ details['connectivity'][1]:.2f }}</li>
                    </ul>
                    <a href="{{ url_for('compare') }}">Сравнить другие изображения</a> | <a href="{{ url_for('index') }}">На главную</a>
                    """,
                    similarity=similarity,
                    interpretation=interpretation,
                    details=details,
                    img_a_b64=img_a_b64,
                    img_b_b64=img_b_b64
                )
            except Exception as e:
                return render_template_string(
                    """
                    <h1>Ошибка</h1>
                    <p>{{ error }}</p>
                    <a href="{{ url_for('compare') }}">Вернуться</a>
                    """,
                    error=str(e)
                )
    return render_template_string(
        """
        <h1>Сравнение двух образцов</h1>
        <p>Загрузите два изображения (спорный документ и образец) для сравнения.</p>
        <form method="post" enctype="multipart/form-data">
            <label>Спорное изображение: <input type="file" name="file_a" accept="image/*"></label><br><br>
            <label>Образец для сравнения: <input type="file" name="file_b" accept="image/*"></label><br><br>
            <input type="submit" value="Сравнить">
        </form>
        <p><a href="{{ url_for('index') }}">На главную</a></p>
        """
    )