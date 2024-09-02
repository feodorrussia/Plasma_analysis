import shtRipper_entry
ripper = shtRipper_entry.Ripper()

import math


to_pack = {
    'signal with error': {
        'comment': 'Точки хранят ещё и погрешность, требует много памяти.',
        'unit': 'T_e(eV)',
        'x': [0, 1, 2],
        'y': [2, 1, 0],
        'err': [1, 2, 1]
    },
    'X Y signal': {
        'comment': 'Точки хранятся парой значений, требует много памяти.',
        'unit': 'T_e(eV)',
        'x': [0, 1, 2],
        'y': [2, 1, 0],
    },
    'Y signal with fixed X-grid': {
        'comment': 'Этот тип хранения данных предназначен для данных 16-бит АЦП',
        'unit': 'U(V)',
        'tMin': 0.0,
        'tMax': 100.0,
        'offset': 0.0,  # ADC zero level offset
        'yRes': 0.0001,  # ADC resolution: 0.0001 Volt per adc bit
        'y': [math.sin(i / 10) for i in range(100)]
    }
}

packed = ripper.write(path='D:/tmp/', filename='TS_test.SHT', data=to_pack)
if len(packed) != 0:
    print('packed error = "%s"' % packed)


adc_data = []
with open('D:/tmp/slow/in.slow', 'rb') as file:
    raw = file.read()
    adc_data.append({
        'bin': raw,
        'unit': 'U(V)',
        'tMin': 0.0,
        'tMax': 100.0,
        'offset': 0.0,  # ADC zero level offset
        'yRes': 20 / 2**16,  # ADC resolution: Volt per adc bit = max voltage / bit count
        'header': [
            {
                'ch': 0,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 1,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 2,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 3,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 4,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 5,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 6,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 7,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 8,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 9,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 10,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 11,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 12,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 13,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 14,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            },
            {
                'ch': 15,
                'name': 'Полихроматор №X, спектральный канал Y',
                'comment': 'АЦП №X, канал #Y'
            }
        ]
    })
packed = ripper.write(path='D:/tmp/', filename='adc_test.SHT', data=adc_data)
if len(packed) != 0:
    print('packed error = "%s"' % packed)

for i in range(50000000):  # wait for possible errors in dll
    d = 56784678 / 5423621543

print('OK.')