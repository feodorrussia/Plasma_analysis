import struct  # built-in library
from pathlib import Path  # built-in library
import shtRipper  # shtRipper-cpp  https://pypi.org/project/shtRipper-cpp/

shots: list[int] = [i for i in range(42587, 42599)]
signal_name_sht: str = 'Ipf2 (7CC) (инт.29)'
#signal_name_sht: str = 'Ip новый (Пр1ВК) (инт.16)' # debug. signal to compare with original .dat
signal_name_dat: str = 'Ipf2'
encoding_niifa: str = 'utf-8'
resample_window: int = 3  # 3 + 1 + 3 rolling averaging is used to resample sht signal to dat.
dat_delay_s: float = 0.14e-3  # 0.2 ms delay between sht and dat
scale: float = 1e-3  # multiplier to get rid of kA.

# paths
path_sht: Path = Path('//172.16.12.28/Data/')  # Globus-3
path_dat_in: Path = Path('//172.16.12.127/Pub/!!!CURRENT_COIL_METHOD/magn_data/') # pub
path_dat_out: Path = Path('//172.16.12.127/Pub/!!!CURRENT_COIL_METHOD/magn_data/pf2_from_sht/') # pub

for shotn in shots:
    print(shotn)

    # get signal from sht file
    signal: dict[str, any] = shtRipper.ripper.read('%ssht%05d.SHT' % (path_sht, shotn), [signal_name_sht])[signal_name_sht]

    # change signal integration time
    signal_integrated: list[float] = [0 for i in range(len(signal['y']))]
    window: int = resample_window * 2 + 1
    rolling_ave: float = sum(signal['y'][: window])
    for time_ind in range(resample_window, len(signal['y']) - resample_window - 1):
        signal_integrated[time_ind] = rolling_ave * scale / window
        rolling_ave += signal['y'][time_ind + resample_window + 1] - signal['y'][time_ind - resample_window]

    # get the original .dat file
    dat: bytes = bytes()
    with open('%s/%08d.dat' % (path_dat_in, shotn), 'rb') as file:
        dat = file.read()

    # write new .dat file
    with open('%s/%08d.dat' % (path_dat_out, shotn), 'wb') as file:
        start: int = dat.find(bytes('t_ms', encoding=encoding_niifa))
        stop: int = dat.find(bytes('\n', encoding=encoding_niifa), start)

        names: list[str] = [entry.decode('utf-8') for entry in dat[start: stop].split(bytes(' ', encoding=encoding_niifa))]
        signal_ind: int = names.index(signal_name_dat)
        stop += 1
        file.write(dat[: stop])

        slice_format: str = '%df' % len(names)
        slice_size: int = 4 * len(names)

        sht_index: int = 0
        for time_ind in range(int((len(dat) - stop)/4/len(names))):
            time_slice: list[float] = list(struct.unpack(slice_format, dat[stop: stop + slice_size]))
            time: float = time_slice[0] * 1e-3 - dat_delay_s
            if time < 0:
                time_slice[signal_ind] = 0  # override signal with 0 to eliminate time shift
            else:
                # find the requested time in integrated signal
                while signal['x'][sht_index] < time:
                    sht_index += 1
                # now sht_index is the index of the point right after the requested time. So time >= signal['x'][sht_index - 1]
                time_slice[signal_ind] = signal_integrated[sht_index - 1] + (signal_integrated[sht_index] - signal_integrated[sht_index - 1]) * (time - signal['x'][sht_index - 1]) / (signal['x'][sht_index] - signal['x'][sht_index - 1])  # override signal with interpolated data from integrated signal
            file.write(struct.pack(slice_format, *time_slice))
            stop += slice_size

print('Code OK')
