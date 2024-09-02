import shtRipper_entry
ripper = shtRipper_entry.Ripper()

import json

print('reading...')
filename = 'd:/tmp/sht41055.SHT'

#res = ripper.read(filename)
res = ripper.read(filename, ['Лазер', 'Emission electrode voltage', 'Emission electrode current'])
print('read OK')


#for i in range(50000000):  # wait for possible errors in dll
#    d = 56784678 / 5423621543

print('OK.')