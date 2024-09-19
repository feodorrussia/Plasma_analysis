import numpy as np
from scipy import signal
import copy
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import time

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

    def copy(self):
        new_slice = Slice(self.l, self.r)
        new_slice.set_mark(new_slice.mark)
        return new_slice

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
    proc_slice = cur_slice.copy()

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
                
                if meta.name == "sxr":
                    if abs(abs(data_diff[proc_slice.l:proc_slice.r].min()) - meta.d_q) > meta.d_std_top * meta.d_std:
                        proc_slice.set_mark(1)
                        
                    if abs(data_diff[proc_slice.l:proc_slice.r].max() / data_diff[proc_slice.l:proc_slice.r].min()) < meta.max_min_ratio:
                        proc_slice.set_mark(1)
                    else:
                        proc_slice.set_mark(0)
                
                res_mark[proc_slice.l: proc_slice.r] = proc_slice.mark
                c += proc_slice.mark
                
                proc_slice = cur_slice.copy()
            f_fragment = False
            cur_slice.collapse_boarders()
        elif not f_fragment:
            cur_slice.collapse_boarders()
            if proc_slice.is_null():
                proc_slice = cur_slice.copy()
    
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
    start_time = time.time()
    
    df = read_sht_data(filename, dir_path)
    df = df.rename(columns={"ch1": "d_alpha"})
    # logg 1
    print("-", end="")
    df["sxr"] = read_sht_data(filename, dir_path, data_name="SXR 50 mkm").ch1
    # mgd_data_vertical
    df["mgd_v"] = read_sht_data(filename, dir_path, data_name="МГД быстрый зонд верт.").ch1
    # mgd_data_radial
    df["mgd_r"] = read_sht_data(filename, dir_path, data_name="МГД быстрый зонд рад.").ch1
    # logg 2
    print("-", end="")
    
    
    d_alpha = df.d_alpha.to_numpy()
    sxr = df.sxr.to_numpy()
    
    d_alpha_d1 = np.diff(d_alpha)
    sxr_d1 = np.diff(sxr)
    
    b, a = signal.butter(5, 0.05)
    d_alpha_f = signal.filtfilt(b, a, d_alpha_d1)
    sxr_f = signal.filtfilt(b, a, sxr_d1)

    mgd = df.mgd_v.to_numpy() ** 2 + df.mgd_r.to_numpy() ** 2
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
    
    sxr_slices = [Slice(0, 0)] + get_slices(mark_sxr)

    sxr_deltas = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    
    deltas_sync_elm = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    amplitudes_sync_mgd = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    deltas_sync_mgd = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    
    count_desync_elm = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    deltas_desync_elm = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    amplitudes_desync_mgd = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    deltas_desync_mgd = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    amplitudes_desync_mgd_2 = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    deltas_desync_mgd_2 = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    deltas_desync_mgd_3 = np.full(shape=len(sxr_slices) - 1, fill_value=np.nan)
    
    report_lines = []
    
    report_lines.append("\n----------------\n")
    report_lines.append(f"Signal {filename}\n")
    report_lines.append(f"SXR falls: {len(sxr_slices) - 1}\n")
    report_lines.append("-----\n")
    l_shift, r_shift = 100, 1500
    # logg 6
    print("-", end="")
                
    # a = 15
    # b = -5
    mgd_std_coef = 1  # a * np.exp(b * mgd.std())
    # print(f"{mgd_std_coef:3.3f}", end="")
    
    for sl_i in range(1, len(sxr_slices)):
        sxr_pointer = np.argmin(sxr_f[sxr_slices[sl_i].l: sxr_slices[sl_i].r]) + sxr_slices[sl_i].l
        sxr_deltas[sl_i - 1] = (sxr_pointer - sxr_slices[sl_i].l) / 1e3
        
        da_slices = get_slices(mark_d_alpha[sxr_slices[sl_i - 1].r: min(sxr_slices[sl_i].r + r_shift, mark_d_alpha.shape[0])])
        mgd_slice = sxr_slices[sl_i].copy()
        mgd_slice.expand(l_shift)
    
        if len(da_slices) == 0:
            deltas_sync_elm[sl_i - 1] = np.nan
            sync_elm_info = f"-- No sync ELM on D-alpha"
            desync_elm_info = f"-- No desync ELM on D-alpha"  # + " " * 15
            
            deltas_sync_mgd[sl_i - 1] = (np.argmax(mgd[mgd_slice.l: mgd_slice.r]) + mgd_slice.l - sxr_pointer) / 1e3
            amplitudes_sync_mgd[sl_i - 1] = abs(mgd[mgd_slice.l: mgd_slice.r].max() - mgd.mean())
            
            if deltas_sync_mgd[sl_i - 1] >= 0.5:
                deltas_sync_mgd[sl_i - 1] = np.nan
                sync_mgd_info = f"-- No nearest peaks on MGD"
            else:
                sync_mgd_info = f"-- MGD peak: delta = {deltas_sync_mgd[sl_i - 1]:3.3f} ms, amplitude = {amplitudes_sync_mgd[sl_i - 1]:3.3f}"
        else:
            ind = 0
            while sxr_slices[sl_i].r - (da_slices[ind].r + sxr_slices[sl_i - 1].l - l_shift) > 0 and ind + 1 < len(da_slices):
                ind += 1
    
            sync_elm_slice = da_slices[ind].copy()
            sync_elm_slice.move(sxr_slices[sl_i - 1].l)
            deltas_sync_elm[sl_i - 1] = (np.argmax(d_alpha_f[sync_elm_slice.l: sync_elm_slice.r]) + sync_elm_slice.l - sxr_pointer) / 1e3
            
            if deltas_sync_elm[sl_i - 1] >= 0.5 or deltas_sync_elm[sl_i - 1] < - l_shift / 1e3:  #  or deltas_sync_elm[sl_i] < - l_shift / 1e3
                deltas_sync_elm[sl_i - 1] = np.nan
                sync_elm_info = f"-- No sync ELM on D-alpha"

                deltas_sync_mgd[sl_i - 1] = (np.argmax(mgd[mgd_slice.l: mgd_slice.r]) + mgd_slice.l - sxr_pointer) / 1e3
                amplitudes_sync_mgd[sl_i - 1] = mgd[mgd_slice.l: mgd_slice.r].max()
            else:
                sync_elm_info = f"-- Sync ELM: d = {deltas_sync_elm[sl_i - 1]:3.3f} ms"

                deltas_sync_mgd[sl_i - 1] = (np.argmax(mgd[mgd_slice.l: mgd_slice.r]) - np.argmax(d_alpha_f[sync_elm_slice.l: sync_elm_slice.r]) + mgd_slice.l - sync_elm_slice.l) / 1e3
                amplitudes_sync_mgd[sl_i - 1] = mgd[mgd_slice.l: mgd_slice.r].max()
                
            if deltas_sync_mgd[sl_i - 1] >= 0.5:
                deltas_sync_mgd[sl_i - 1] = np.nan
                sync_mgd_info = f"-- No nearest peaks on MGD"
            else:
                sync_mgd_info = f"-- MGD peak: delta = {deltas_sync_mgd[sl_i - 1]:3.3f} ms, amplitude = {amplitudes_sync_mgd[sl_i - 1]:3.3f}"
    
            if ind - 1 <= 0:
                count_desync_elm[sl_i - 1] = np.nan
                desync_elm_info = f"-- No desync ELM on D-alpha"  # + " " * 15
                desync_mgd_info = ""
            else:
                count_desync_elm[sl_i - 1] = ind - 1
                
                cur_deltas_desync_elm = np.zeros(int(count_desync_elm[sl_i - 1] - 1))
                cur_desync_mgd_amplitude = np.zeros(int(count_desync_elm[sl_i - 1]))
                cur_desync_mgd_deltas = np.zeros(int(count_desync_elm[sl_i - 1]))
                cur_desync_mgd_amplitude_2 = np.zeros(int(count_desync_elm[sl_i - 1]))
                cur_desync_mgd_deltas_2 = np.zeros(int(count_desync_elm[sl_i - 1]))
                cur_desync_mgd_deltas_3 = np.zeros(int(count_desync_elm[sl_i - 1]))
                
                desync_elm_slice = da_slices[1].copy()
                desync_elm_slice.move(sxr_slices[sl_i - 1].l)
    
                prev_argmax = np.argmax(d_alpha_f[desync_elm_slice.l: desync_elm_slice.r]) + desync_elm_slice.l
                
                cur_desync_mgd_amplitude[0] = mgd[desync_elm_slice.l - 100: desync_elm_slice.r + 100].max()
                cur_desync_mgd_deltas[0] = (np.argmax(mgd[desync_elm_slice.l - 100: desync_elm_slice.r + 100]) + desync_elm_slice.l - 100 - prev_argmax) / 1e3

                std_check_arr = np.argwhere(abs(mgd[desync_elm_slice.l - 100: desync_elm_slice.r + 100] - mgd.mean()) >  mgd_std_coef * mgd.std())
                if std_check_arr.shape[0] == 0:
                    cur_desync_mgd_amplitude_2[0] = cur_desync_mgd_amplitude[0]
                    cur_desync_mgd_deltas_2[0] = cur_desync_mgd_deltas[0]
                    cur_desync_mgd_deltas_3[0] = 0.0
                else:
                    cur_desync_mgd_amplitude_2[0] = mgd[std_check_arr[0, 0]]
                    cur_desync_mgd_deltas_2[0] = (std_check_arr[0, 0] + desync_elm_slice.l - 100 - prev_argmax) / 1e3
                    cur_desync_mgd_deltas_3[0] = (np.argmax(mgd[desync_elm_slice.l - 100: desync_elm_slice.r + 100]) - std_check_arr[0, 0]) / 1e3
                
                for elm_ind in range(2, ind - 1):
                    desync_elm_slice = da_slices[elm_ind].copy()
                    desync_elm_slice.move(sxr_slices[sl_i - 1].l)
                    
                    cur_argmax = np.argmax(d_alpha_f[desync_elm_slice.l: desync_elm_slice.r]) + desync_elm_slice.l
                    cur_deltas_desync_elm[elm_ind - 1] = (cur_argmax - prev_argmax) / 1e3
                    
                    cur_desync_mgd_amplitude[elm_ind] = mgd[desync_elm_slice.l - 100: desync_elm_slice.r + 100].max()
                    cur_desync_mgd_deltas[elm_ind] = (np.argmax(mgd[desync_elm_slice.l - 100: desync_elm_slice.r + 100]) + desync_elm_slice.l - 100 - cur_argmax) / 1e3

                    # print(np.argwhere(abs(mgd[desync_elm_slice.l - 100: desync_elm_slice.r + 100] - mgd.mean()) >  mgd.std()))
                    std_check_arr = np.argwhere(abs(mgd[desync_elm_slice.l - 100: desync_elm_slice.r + 100] - mgd.mean()) > mgd_std_coef * mgd.std())
                    if std_check_arr.shape[0] == 0:
                        cur_desync_mgd_amplitude_2[elm_ind] = cur_desync_mgd_amplitude[elm_ind]
                        cur_desync_mgd_deltas_2[elm_ind] = cur_desync_mgd_deltas[elm_ind]
                        cur_desync_mgd_deltas_3[elm_ind] = 0.0
                    else:
                        cur_desync_mgd_amplitude_2[elm_ind] = mgd[std_check_arr[0, 0]]
                        cur_desync_mgd_deltas_2[elm_ind] = (std_check_arr[0, 0] + desync_elm_slice.l - 100 - cur_argmax) / 1e3
                        cur_desync_mgd_deltas_3[elm_ind] = (np.argmax(mgd[desync_elm_slice.l - 100: desync_elm_slice.r + 100]) - std_check_arr[0, 0]) / 1e3
                    
                    prev_argmax = cur_argmax
    
                deltas_desync_elm[sl_i - 1] = cur_deltas_desync_elm.mean() if count_desync_elm[sl_i - 1] - 1 > 0 else np.nan
                amplitudes_desync_mgd[sl_i - 1] = np.nanmean(cur_desync_mgd_amplitude)
                deltas_desync_mgd[sl_i - 1] = np.nanmean(cur_desync_mgd_deltas)
                amplitudes_desync_mgd_2[sl_i - 1] = np.nanmean(cur_desync_mgd_amplitude_2)
                deltas_desync_mgd_2[sl_i - 1] = np.nanmean(cur_desync_mgd_deltas_2)
                deltas_desync_mgd_3[sl_i - 1] = np.nanmean(cur_desync_mgd_deltas_3)
                
                desync_elm_info = f"-- Desync ELM: count = {count_desync_elm[sl_i - 1]}"
                if count_desync_elm[sl_i - 1] > 1 and cur_deltas_desync_elm.mean() > 1e-6:
                    desync_elm_info += f", fr mean = {1 / cur_deltas_desync_elm.mean():3.3f} kGz, fr std = {cur_deltas_desync_elm.std() / (cur_deltas_desync_elm.mean() ** 2):3.3f} kGz"  # d mean = {desync_elm_deltas.mean():3.3f}, d std = {desync_elm_deltas.std():3.3f} ms, 
                else:
                    desync_elm_info += f", fr mean = nan kGz, fr std = nan kGz"
                desync_mgd_info = f"-- MGD peaks: deltas mean = {np.nanmean(cur_desync_mgd_deltas):3.3f} ms, deltas std = {np.nanstd(cur_desync_mgd_deltas):3.3f} ms, "
                desync_mgd_info += f"A mean = {np.nanmean(cur_desync_mgd_amplitude):6.6f}, A std = {np.nanstd(cur_desync_mgd_amplitude):6.6f}"
                desync_mgd_info = f"-- first MGD > std: deltas(/MGD peaks) mean = {np.nanmean(cur_desync_mgd_deltas_3):.3e} ms, deltas(/MGD peaks) std = {np.nanstd(cur_desync_mgd_deltas_3):.3e} ms, "
                desync_mgd_info += f"A mean = {np.nanmean(cur_desync_mgd_amplitude_2):.3e}, A std = {np.nanstd(cur_desync_mgd_amplitude_2):.3e}"
            
        
        report_lines.append(f"{sl_i}. SXR fall - {sxr_pointer / 1e3:3.3f} ms - delta = {sxr_deltas[sl_i - 1]}\n\t{sync_elm_info}\n\t{sync_mgd_info}\n\t{desync_elm_info}\n\t{desync_mgd_info}\n")
    
    # logg 8
    print("-", end="")
    
    
    report_lines.append("-----\n")
    report_lines.append(f"SXR d1 peaks info: deltas mean = {np.nanmean(sxr_deltas):.3f} ms, deltas std = {np.nanstd(sxr_deltas):.3f} ms\n")
    report_lines.append(f"Sync ELM info: deltas mean = {np.nanmean(deltas_sync_elm):.3f} ms, deltas std = {np.nanstd(deltas_sync_elm):.3f} ms\n")
    report_lines.append(f"Desync ELM info: count mean = {np.nanmean(count_desync_elm[1:]):.3f}, count std = {np.nanstd(count_desync_elm[1:]):.3f}, " +
                        f"frequency mean = {1 / np.nanmean(deltas_desync_elm[1:]):.3f} ms, frequency std = {np.nanstd(deltas_desync_elm[1:]) / (np.nanmean(deltas_desync_elm[1:]) ** 2):.3f}\n")
    report_lines.append(f"MGD peaks info (sync ELM): deltas mean = {np.nanmean(deltas_sync_mgd):.3f} ms, deltas std = {np.nanstd(deltas_sync_mgd):.3f} ms, " +
                        f"A mean = {np.nanmean(amplitudes_sync_mgd):.3f}, A std = {np.nanstd(amplitudes_sync_mgd):.3f}\n")
    report_lines.append(f"MGD peaks info (desync ELM): deltas mean = {np.nanmean(deltas_desync_mgd[1:]):.3f} ms, " +
                        f"A mean = {np.nanmean(amplitudes_desync_mgd[1:]):.3f}\n")
    report_lines.append("-----\n")
    report_lines.append(f"SXR falls w/o sync ELM in nearest area (-{l_shift * 1e-3} ms; {r_shift * 1e-3} ms): {np.count_nonzero(np.isnan(deltas_sync_elm))}\n")
    report_lines.append(f"SXR falls w/o peaks on MGD in nearest area (-{l_shift * 1e-3} ms; {l_shift * 1e-3} ms): {np.count_nonzero(np.isnan(deltas_sync_mgd))}\n")
    report_lines.append("-----\n")
    report_lines.append(f"SXR signal info: diff_quantile = {meta_sxr.d_q:.6f}, diff_std = {meta_sxr.d_std:.6f}\n")
    report_lines.append(f"MGD signal info: mean = {mgd.mean():.6f}, std = {mgd.std():.6f}\n")
    report_lines.append("-----\n")
    report_lines.append(f"SXR diff_std_top_edge: {meta_sxr.d_std_bottom:.3f} (approximate w/ a*exp^b)\n")
    report_lines.append("----------------\n")
    # logg 9
    print("-", end="")

    report_path = report_filename if "/" in report_filename or "\\" in report_filename else dir_path + report_filename

    with open(report_path, "a") as file:
        file.writelines(report_lines)

    # logg 10
    print("-|", end="")
    print(f" - signal (sxr: d_q={meta_sxr.d_q:.6f}, d_std={meta_sxr.d_std:.6f}, d_std_top_edge={meta_sxr.d_std_bottom:.3f})", end="")
    print(f" - w/ ELM {len(sxr_slices) - np.count_nonzero(np.isnan(deltas_sync_elm))} (m={np.nanmean(deltas_sync_elm):.3f}, std={np.nanstd(deltas_sync_elm):.3f}) | w/o ELM {np.count_nonzero(np.isnan(deltas_sync_elm))}", end="")
    print(f" - {(time.time() - start_time):.4f} s")

    return [
        sxr_deltas,
        deltas_sync_elm,
        deltas_sync_mgd,
        amplitudes_sync_mgd,
        count_desync_elm[1:],
        deltas_desync_elm[1:],
        deltas_desync_mgd[1:],
        amplitudes_desync_mgd[1:],
        deltas_desync_mgd_2[1:],
        deltas_desync_mgd_3[1:],
        amplitudes_desync_mgd_2[1:],
    ]
    

if __name__ == "__main__" and not (sys.stdin and sys.stdin.isatty()):
    # get args from CL
    print("Sys args (dir filepath | name of the report file):\n", sys.argv)
    print("\n----------------")

    report_path = sys.argv[2] if "/" in sys.argv[2] or "\\" in sys.argv[2] else sys.argv[1] + sys.argv[2]

    with open(report_path, "w") as file:
        file.write("The sync ELMs delta was considered relative to the peaks of the first diff values on SXR & D-alpha.\n" +
                   "The MGD peaks delta was considered relative to the peaks of the absolute MGD values & first diff values on SXR (vert^2 + rad^2).\n")

    data = []
    labels = []
    for f_name in os.listdir(sys.argv[1]):
        print(f"Process {f_name} - |", end="")
        data.append(init_app(f_name[:-4], sys.argv[1], sys.argv[2]))
        labels.append(f_name)

    titles = ['SXR d1 peak delta, ms', 'sELM/SXR delta, ms', 'MGD/sELM delta, ms', 'MGD(sELM) peaks A', 'dsELM count', 'dsELM fr, kGz', 'peak MGD/dsELM delta, ms', 'peaks MGD(dsELM) A', 'precur MGD/dsELM delta, ms', 'precur MGD(dsELM) delta delta, ms', 'precur MGD(dsELM) A']

    n_rows=len(data[0])
    fig, axs = plt.subplots(nrows=n_rows)
    
    fig.set_figwidth(16)
    fig.set_figheight(n_rows * 3 + n_rows // 3)
    
    for i in range(n_rows):
        axs[i].set_ylabel(titles[i])
        deltas_data = []
        
        for data_set in data:
            if titles[i] == 'dsELM fr':
                data_arr = (1 / data_set[i][np.nonzero(data_set[i])])
            else:
                data_arr = data_set[i]
            deltas_data.append(data_arr[~ np.isnan(data_arr)])
            
        bplot = axs[i].boxplot(deltas_data, labels=labels)
        axs[i].grid(which='major', color='#DDDDDD', linewidth=0.9)
        axs[i].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.7)
        axs[i].minorticks_on()
        axs[i].xaxis.set_minor_locator(AutoMinorLocator(10))

    # C:/Users/f.belous/Work/Projects/Plasma_analysis | D:/Edu/Lab/Projects/Plasma_analysis
    plt.savefig(f"D:/Edu/Lab/Projects/Plasma_analysis/img/boxplots/{sys.argv[2][-13:-4]}_boxplot.png")

else:
    print("Program is supposed to run out from command line.")
