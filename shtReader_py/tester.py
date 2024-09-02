import python.shtRipper

ripper = python.shtRipper.ripper

import time

filename = '/home/nz/Downloads/sht41384.SHT'


start_time = time.time()
for iteration in range(1):
    #pass
    #res = shtRipper.ripper.read(filename)
    res = ripper.read(filename)
    #print('yeah')
print("--- %.2f seconds ---" % (time.time() - start_time))


for i in range(5000000):  # wait for possible errors in dll
    d = 56784678 / 5423621543

print('OK.')
