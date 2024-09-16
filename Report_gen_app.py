import numpy as np
from scipy import signal
import copy
import sys
import os

from source.Files_operating import read_sht_data


def get_boarders(data: np.array, loc_max_ind=None, scale=1.5):
    loc_max_ind = np.argmax(data)
    dist_ind = np.argsort(np.abs(data - data[loc_max_ind] / scale))
    return Slice(dist_ind[dist_ind <= loc_max_ind][0], dist_ind[dist_ind >= loc_max_ind][0])


class Slice:
    def __init__(self, start_index=0, end_index=0):
        self.l = start_index
        self.r = end_index
        self.mark = 1.0

    def set_boarders(self, start_index: int, end_index: int) -> None:
        self.l = start_index
        self.r = end_index

    def set_mark(self, mark: int) -> None:
        self.mark = mark

    def copy(self, other):
        self.l = other.l
        self.r = other.r
        self.mark = other.mark

    def check_length(self, len_edge: int) -> bool:
        return self.r - self.l > len_edge

    def check_dist(self, other, dist_edge: int) -> bool:
        return other.l - self.r > dist_edge

    def collide_slices(self, other, dist_edge: int) -> bool:
        if not self.check_dist(other, dist_edge):
            self.r = other.r
            return True
        return False

    def step(self) -> None:
        self.r += 1

    def move(self, delta: int) -> None:
        self.r += delta
        self.l += delta

    def expand(self, delta: int) -> None:
        self.r += delta
        self.l -= delta

    def collapse_boarders(self) -> None:
        self.l = self.r

    def is_null(self) -> bool:
        return self.l == self.r

    def __repr__(self):
        return f"({self.l}, {self.r})"


class Signal_meta:
    def __init__(self, chanel_name="da", processing_flag=False, 
                 quantile_edge=0.0, std_edge=1.0, 
                 length_edge=10, distance_edge=10, scale=1.5, step_out=10, 
                 std_bottom_edge=0, std_top_edge=1, d_std_bottom_edge=3, d_std_top_edge=6, amplitude_ratio=0.5):
        self.name = chanel_name
        self.proc_fl = processing_flag
        
        self.len_edge = length_edge
        self.dist_edge = distance_edge
        self.scale = scale
        self.step_out = step_out
        
        self.q = quantile_edge
        self.std = std_edge
        
        self.d_q = quantile_edge
        self.d_std = std_edge

        self.std_top = std_top_edge
        self.std_bottom = std_bottom_edge
        self.d_std_top = d_std_top_edge
        self.d_std_bottom = d_std_bottom_edge

        self.max_min_ratio = amplitude_ratio
        
    def set_statistics(self, data: np.array, data_diff: np.array, percentile: float, d_percentile: float, std_bottom_edge=0, std_top_edge=1, d_std_bottom_edge=3, d_std_top_edge=6, amplitude_ratio=0.5):
        self.q = np.quantile(data, percentile)
        self.std = data.std()
        self.std_top = std_top_edge
        self.std_bottom = std_bottom_edge
        
        self.d_q = np.quantile(data_diff, percentile)
        self.d_std = data_diff.std()
        if self.name == "sxr":
            a = 10.5
            b = -850
            self.d_std_top = d_std_top_edge / d_std_bottom_edge * a * np.exp(b * self.d_std)
            self.d_std_bottom = a * np.exp(b * self.d_std)
        else:
            self.d_std_top = d_std_top_edge
            self.d_std_bottom = d_std_bottom_edge            

        self.max_min_ratio = amplitude_ratio

    def set_edges(self, length_edge=10, distance_edge=10, scale=1.5, step_out=10):
        self.len_edge = length_edge
        self.dist_edge = distance_edge
        self.scale = scale
        self.step_out = step_out


def get_peaks(data: np.array, s_i: int) -> np.array:
    peaks_ind = []
    loc_max = data.min()
    m_v = data.mean()
    loc_max_ind = 0
    increase_fl = False
    for i in range(data.shape[0] - 1):
        if loc_max < data[i]:
            increase_fl = True
            loc_max = data[i]
            loc_max_ind = i
        elif abs((loc_max - data[i]) / (loc_max + 1e-10)) > 0.5 and increase_fl:
            if abs(loc_max) > 5 * abs(m_v + 1e-10):
                peaks_ind.append(loc_max_ind)
            # print(s_i + loc_max_ind, abs((loc_max - data[i]) / loc_max), abs((loc_max - m_v) / m_v))  # len(peaks_ind),
            increase_fl = False

        if not increase_fl or data[i] < data[i + 1]:
            loc_max = data[i]
            loc_max_ind = i
    return np.array(peaks_ind)

def get_boarders_d2(data:np.array, diff_data: np.array, s_i: int, scale=1.5):
    d2_data = np.diff(diff_data)
    peaks_ind = get_peaks(d2_data, s_i)

    if len(peaks_ind) == 0:
        return Slice(0, diff_data.shape[0])
    if len(peaks_ind) == 1:
        scale_slice = get_boarders(data, scale=scale)
        # print(scale_slice)
        if peaks_ind[0] < diff_data.shape[0] - peaks_ind[0]:
            if peaks_ind[0] - scale_slice.r < 0:
                return Slice(peaks_ind[0], scale_slice.r)
            else:
                return Slice(peaks_ind[0], diff_data.shape[0])
        else:
            if peaks_ind[0] - scale_slice.l > 0:
                return Slice(scale_slice.l, peaks_ind[0])
            else:
                return Slice(0, peaks_ind[0])
    return Slice(peaks_ind[0], peaks_ind[-1])

def proc_boarders(data: np.array, data_diff: np.array, start_ind: int, scale=1.5) -> Slice:
    step = 5
    
    diff_coeff = 1
    if data_diff[start_ind] < 0:
        diff_coeff = -1
    
    cur_ind = (start_ind + diff_coeff * step) if 0 < start_ind + diff_coeff * step < data.shape[0] else start_ind
    while 0 < cur_ind + diff_coeff * step < data.shape[0] and data_diff[cur_ind] * data_diff[cur_ind + diff_coeff * step] > 0:
        cur_ind += diff_coeff * step

    # print(cur_ind - 3 * step, cur_ind + 3 * step, end=" ")
    max_ind = np.argmax(data[max(cur_ind - 3 * step, 0): min(cur_ind + 3 * step, data.shape[0])]) + cur_ind - step
    length = max(abs(max_ind - start_ind), 3 * step)
    # print(max_ind, length)

    # print(max_ind - length, max_ind + 2 * length, end=" ")
    res_slice = get_boarders_d2(data[max(max_ind - length, 0): min(max_ind + 2 * length, data.shape[0])], data_diff[max(max_ind - length, 0): min(max_ind + 2 * length, data.shape[0])], max(max_ind - length, 0), scale=scale)  # get_boarders(data[max_ind - length: max_ind + 2 * length], loc_max_ind=length) | Slice(0, 3 * length)
    res_slice.move(max_ind - length)
    # print(res_slice.l, res_slice.r)

    # add checking diff on right & left boarder (cut on D2 peaks) - done
    # add dtw classification (None | ELM | LSO)

    # print(res_slice.l, res_slice.r, end=" ")
    # print(abs(data_diff[res_slice.l:res_slice.r].max() - np.quantile(data_diff, 0.7)), data_diff.std())
    
    return res_slice


def proc_slices(mark_data: np.array, data: np.array, data_diff: np.array, meta: Signal_meta) -> np.array:  # data: np.array, , scale=1.5 , step_out=10
    proc_slice = Slice(0, 50)
    cur_slice = Slice(0, 51)
    f_fragment = False

    res_mark = np.copy(mark_data)
    res_mark[cur_slice.l: cur_slice.r] = 0.0
    cur_slice.collapse_boarders()
    proc_slice.copy(cur_slice)

    c = 0
    
    while cur_slice.r < res_mark.shape[0]:
        if res_mark[cur_slice.r] == 1.0:
            if not f_fragment:
                f_fragment = True
        elif f_fragment:
            # print(start_ind, end_ind)
            if not cur_slice.check_length(meta.len_edge):
                res_mark[cur_slice.l: cur_slice.r] = 0.0
            elif not proc_slice.collide_slices(cur_slice, meta.dist_edge):
                if meta.proc_fl and meta.scale > 1:
                    res_mark[proc_slice.l: proc_slice.r] = 0.0
                    start_ind = proc_slice.l if data_diff[proc_slice.l] > 0 else proc_slice.r
                    proc_slice = proc_boarders(data, data_diff, start_ind, meta.scale)
                    
                    cur_slice = Slice(proc_slice.r, proc_slice.r)
                
                proc_slice.expand(meta.step_out)

                if meta.proc_fl and abs(data_diff[proc_slice.l:proc_slice.r].max() - meta.d_q) < meta.d_std_top * meta.d_std and \
                   abs(data_diff[proc_slice.l:proc_slice.r].min() - meta.d_q) < meta.d_std_top * meta.d_std:
                    proc_slice.set_mark(0)

                if meta.name == "sxr" and abs(data_diff[proc_slice.l:proc_slice.r].max() / data_diff[proc_slice.l:proc_slice.r].min()) < meta.max_min_ratio:
                    proc_slice.set_mark(1)

                if meta.name == "sxr" and abs(data_diff[proc_slice.l:proc_slice.r].max() - meta.d_q) > meta.d_std_top * meta.d_std:
                    proc_slice.set_mark(0)
                
                res_mark[proc_slice.l: proc_slice.r] = proc_slice.mark
                c += proc_slice.mark
                
                proc_slice.copy(cur_slice)
            f_fragment = False
            cur_slice.collapse_boarders()
        elif not f_fragment:
            cur_slice.collapse_boarders()
            if proc_slice.is_null():
                proc_slice.copy(cur_slice)
    
        cur_slice.step()
    # print(c)

    return res_mark


def get_slices(mark_data: np.array):
    cur_slice = Slice(0, 1)
    f_fragment = False

    slices_list = []
    
    while cur_slice.r < mark_data.shape[0]:
        if mark_data[cur_slice.r] == 1.0:
            if not f_fragment:
                f_fragment = True
        elif f_fragment:
            slices_list.append(copy.copy(cur_slice))
            f_fragment = False
            cur_slice.collapse_boarders()
        elif not f_fragment:
            cur_slice.collapse_boarders()
        cur_slice.step()

    return slices_list


def init_app(filename, dir_path, report_filename=""):    
    df = read_sht_data(filename, dir_path)
    df = df.rename(columns={"ch1": "d_alpha"})
    # logg 1
    print("-", end="")
    df["sxr"] = read_sht_data(filename, dir_path, data_name="SXR 50 mkm").ch1
    # logg 2
    print("-", end="")
    
    
    d_alpha = df.d_alpha.to_numpy()
    sxr = df.sxr.to_numpy()
    
    d_alpha_d1 = np.diff(d_alpha)
    sxr_d1 = np.diff(sxr)
    
    b, a = signal.butter(5, 0.05)
    d_alpha_f = signal.filtfilt(b, a, d_alpha_d1)
    sxr_f = signal.filtfilt(b, a, sxr_d1)
    # logg 3
    print("-", end="")
    
    meta_da = Signal_meta(chanel_name="da", processing_flag=True)
    meta_da.set_statistics(d_alpha, d_alpha_f, 0.7, 0.7, d_std_bottom_edge=1.5, d_std_top_edge=2.7)
    meta_da.set_edges(length_edge=15, distance_edge=30)
    
    meta_sxr = Signal_meta(chanel_name="sxr", processing_flag=True)
    meta_sxr.set_statistics(sxr, sxr_f, 0.8, 0.8, d_std_bottom_edge=7.0, d_std_top_edge=15.0)
    meta_sxr.set_edges(length_edge=10, distance_edge=30, scale=0, step_out=20)
    
    mark_data = np.zeros(d_alpha_f.shape)
    mark_data[abs(d_alpha_f - meta_da.d_q) > meta_da.d_std * meta_da.d_std_bottom] = 1
    mark_d_alpha = proc_slices(mark_data, d_alpha, d_alpha_f, meta_da)
    # logg 4
    print("-", end="")
    
    mark_data = np.zeros(d_alpha_f.shape)
    mark_data[abs(sxr_f - meta_sxr.d_q) > meta_sxr.d_std * meta_sxr.d_std_bottom] = 1
    mark_sxr = proc_slices(mark_data, sxr, sxr_f, meta_sxr)
    # logg 5
    print("-", end="")
    
    sxr_slices = get_slices(mark_sxr)
    deltas = np.zeros(len(sxr_slices))
    # logg 6
    print("-", end="")

    report_lines = []
    
    report_lines.append("\n----------------\n")
    report_lines.append(f"Signal {filename}\n")
    report_lines.append(f"SXR falls: {len(sxr_slices)}\n")
    report_lines.append("-----\n")
    l_shift, r_shift = 100, 1500
    
    for sl_i in range(len(sxr_slices)):
        da_slices = get_slices(mark_d_alpha[min(sxr_slices[sl_i].l - l_shift, mark_d_alpha.shape[0]): min(sxr_slices[sl_i].r + r_shift, mark_d_alpha.shape[0])])
    
        if len(da_slices) == 0:
            deltas[sl_i] = np.nan
            report_lines.append(f"SXR fall - {(((sxr_slices[sl_i].r - sxr_slices[sl_i].l) // 2 + sxr_slices[sl_i].l) / 1e3):3.3f} ms -- No ELM on D-alpha\n")
        else:
            ind = 0
            while sxr_slices[sl_i].r - (da_slices[ind].r + sxr_slices[sl_i].l - l_shift) > 0 and ind + 1 < len(da_slices):
                ind += 1
            
            deltas[sl_i] = (da_slices[ind].l - l_shift) / 1e3
            
            if deltas[sl_i] > 0.4:
                deltas[sl_i] = np.nan
                report_lines.append(f"SXR fall - {(((sxr_slices[sl_i].r - sxr_slices[sl_i].l) // 2 + sxr_slices[sl_i].l) / 1e3):3.3f} ms -- No ELM on D-alpha\n")
                continue
            report_lines.append(f"SXR fall - {(((sxr_slices[sl_i].r - sxr_slices[sl_i].l) // 2 + sxr_slices[sl_i].l) / 1e3):3.3f} ms -- ELM on D-alpha: delta = {deltas[sl_i]:3.3f} ms\n")
    # logg 8
    print("-", end="")
    
    
    report_lines.append("-----\n")
    report_lines.append(f"Deltas info: mean = {np.nanmean(deltas):.3f} ms, std = {np.nanstd(deltas):.3f}\n")
    report_lines.append(f"SXR falls w/o sync ELM in nearest area (-{l_shift * 1e-3} ms; {r_shift * 1e-3} ms): {np.count_nonzero(np.isnan(deltas))}\n")
    report_lines.append(f"SXR info: diff_quantile = {meta_sxr.d_q:.6f}, diff_std = {meta_sxr.d_std:.6f}\n")
    report_lines.append(f"SXR diff_std_top_edge: {meta_sxr.d_std_bottom:.3f} (approximate w/ a*exp^b)\n")
    # logg 9
    print("-", end="")

    report_path = report_filename if "/" in report_filename or "\\" in report_filename else dir_path + report_filename

    with open(report_path, "a") as file:
        file.writelines(report_lines)

    # logg 10
    print("-|", end="")
    print(f" - signal (sxr: d_q={meta_sxr.d_q:.6f}, d_std={meta_sxr.d_std:.6f}, d_std_top_edge={meta_sxr.d_std_bottom:.3f})", end="")
    print(f" - w/ ELM {len(sxr_slices) - np.count_nonzero(np.isnan(deltas))} (m={np.nanmean(deltas):.3f}, std={np.nanstd(deltas):.3f}) | w/o ELM {np.count_nonzero(np.isnan(deltas))}")

if __name__ == "__main__" and not (sys.stdin and sys.stdin.isatty()):
    # get args from CL
    print("Sys args (dir filepath | name of the report file):\n", sys.argv)
    print("\n----------------")
    for f_name in os.listdir(sys.argv[1]):
        print(f"Process {f_name} - |", end="")
        init_app(f_name[:-4], sys.argv[1], sys.argv[2])

else:
    print("Program is supposed to run out from command line.")
