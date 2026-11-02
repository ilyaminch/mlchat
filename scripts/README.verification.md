# Локальные геоданные для проверок MLChat

Источник геоданных — соседний репозиторий backend, `data/verification`.
После получения обоих репозиториев выполните из корня MLChat:

```powershell
python scripts/sync_backend_verification.py
```

Для другого расположения backend есть `--backend-dir ПУТЬ`. Скрипт проверяет
SHA-256 по backend/source-reference.json и, если установлены локальные
`fixtures/mlchat/dataset-reference.json`, также по независимому эталону MLChat.
При несовпадении он отказывает до копирования. Ожидания автоматически не меняются.
Игнорируемый каталог MLChat `data/verification` — только локальная копия для
существующих инженерных сценариев, не вторая поставка и не часть wheel.

Это 28 объектов и 4 района, не полный демонабор 500+4. У некоторых объектов
поля/геометрия искусственные; происхождение и атрибуция остаются в sources.json
и README backend. Fixtures MLChat поставляются его разработчиком отдельно.
Архивный filter_validation использует прежний AST 1.0 и не подключается к
runtime MLChat: действующий контракт — native FilterV1 установленного backend.

Рабочий MLChat получает capabilities и исполняет поиск через backend. Читать
локальные GeoJSON внутри интерпретатора для получения результатов не требуется.
Импортные файлы frontend находятся в backend/data/verification/imports, а не
в исходном смешанном objects.geojson.
