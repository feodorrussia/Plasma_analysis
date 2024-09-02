# shtRipper

На винде в зависимостях MSVS140

Код для извлечения данных из .sht файлов. На текущий момент файлы версии ниже 2 не поддерживаются, что будет исправлено при необходимости.

Перед обновлением пакета остановить все запущенные питоновские интерпретаторы!
Надёжнее удалять предыдущую версию, чтобы гарантированно обновились бинарные файлы.


Пример кода для чтения всего sht файла:

    import shtRipper
    
    filename = 'd:/data/cfm/original/sht40808.SHT'
    res = shtRipper.ripper.read(filename)
    print(res.keys())

---   

Пример кода для чтения только выбранных сигналов:

    import shtRipper
    
    filename = 'd:/data/cfm/original/sht40808.SHT'
    res = shtRipper.ripper.read(filename, ['Лазер', 'Emission electrode voltage', 'Emission electrode current'])
    print(res.keys())

---


Пример кода для упаковки данных в формат .sht:

    import shtRipper
    import math  # used only for example sin() signal

    to_pack = {
       'signal with error': {
           'comment': 'Точки хранят ещё и погрешность, требует много памяти.',
           'unit': 'T_e(eV)',
           'x': x,
           'y': T_c,
           'err': T_err
       },
       'X Y signal': {
           'comment': 'Точки хранятся парой значений, требует много памяти.',
           'unit': 'T_e(eV)',
           'x': x,
           'y': n_c
       },
       'Y signal with fixed X-grid': {
           'comment': 'Этот тип хранения данных предназначен для данных 16-бит АЦП',
           'unit': 'U(V)',
           'tMin': 0.0,  # mininun time
           'tMax': 100.0,  # maximum time
           'offset': 0.0,  # ADC zero level offset
           'yRes': 0.0001,  # ADC resolution: 0.0001 Volt per adc bit
           'y': [math.sin(i / 10) for i in range(100)]
       }
    }
    
    packed = shtRipper.ripper.write(path='D:/tmp/', filename='example.SHT', data=to_pack)
    if len(packed) != 0:
        print('packed error = "%s"' % packed)

---   