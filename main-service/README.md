Open three new consoles and follow the commands:
1. In all consoles type: ```cd backend```
2. In all consoles type: ```python -m venv .venv```
3. In all consoles type: ```.venv/Scripts/activate```
4. In one console type: ```pip install -r configs/requirements.txt```
6. In the first console add this command -> ```python -m clientService.app``` -> and in the second console add this command -> ```python -m authService.app``` -> and in the third console add this command -> ```python -m modelService.app```
7. Open browser and type this urls: ```http://localhost:8001/docs```, ```http://localhost:8002/docs```, ```http://localhost:8003/docs```


сделайть топ 7 и листать
сделать блок с описанием парсинга(сначала развернут)
сделать скачивание файла после ллм
изначально развернуть логотип


**важно**
переименовать дашборд - общий анализ
сделать отдельную вкладку по продуктам подтягивается дата из общий анализ



У меня уже есть алгоритм, который добавляет задачи в APScheduler при нажатии кнопки «Начать парсить». Однако он работает некорректно.

Вот что нужно исправить:

1. Оптимизировать работу APScheduler и проверить логику в HTML и JS.
2. Не показывать пользователю период, за который будет выполняться парсинг. Вместо этого предложить несколько вариантов частоты выполнения: — не выбрано —, Через день, Еженедельно, Ежемесячно или Ежегодно. Если выбрано — не выбрано —, алгоритм сработает один раз. В противном случае задачи будут добавляться в APScheduler регулярно, в соответствии с выбранным интервалом.
3. В зависимости от выбора пользователя, кнопка должна менять своё название. Если выбрано — не выбрано —, она должна называться «Немедленно начать». Если выбран интервал, кнопка должна называться «Начать».
