import numpy as np
import copy


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

    def to_list(self):
        return [self.l, self.r]

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


def get_d2_peaks(diff_data: np.array, mark_data: np.array, s_i: int):
    l_ = 0
    fr_fl = False
    peaks_ind = []
    for i in range(mark_data.shape[0]):
        if mark_data[i] == 1 and not fr_fl:
            fr_fl = True
            l_ = i
        elif fr_fl and mark_data[i] == 0:
            fr_fl = False
            d2_data = np.diff(diff_data)
            res = get_peaks(d2_data[l_-10:i+30], s_i) + l_-10
            peaks_ind += res.tolist()
    peaks_ind = np.array(peaks_ind)
    return peaks_ind, d2_data[peaks_ind]


def zero_bin_search(arr: np.array):
    l = 0
    r = arr.shape[0] - 1
    if arr[l] * arr[r] > 0 or r - l <= 0:
        return -1
    
    while r - l > 1:
        m = (r - l) // 2 + l
        if arr[l] * arr[m] <= 0:
            r = m
        else:
            l = m
    
    return r if arr[l] > arr[r] else l


def get_d1_crosses(d1_data: np.array, d2_data: np.array, start: int, end: int, d1_coef=1) -> np.array:
    d1_std = d1_data.std()
    d1_m = d1_data.mean()
    d2_std = d2_data.std()
    d2_m = d2_data.mean()
    
    cur_slice = Slice(start, start + 1)
    f_slice = False
    ans = []
    while cur_slice.r < end + 1:
        if abs(d1_data[cur_slice.r] - d1_m) < d1_std * d1_coef and d2_data[cur_slice.r] - d2_m < d2_std / 3 * 2:
            if not f_slice and (d1_data[cur_slice.r - 1] - d1_m) > d1_std * d1_coef:
                f_slice = True
        
        elif f_slice:  #  and d1_data[cur_slice.r] < 0
            # print(cur_slice)
            if d1_data[cur_slice.r] > d1_m:
                cur_slice.r = np.argmin(d1_data[cur_slice.l: cur_slice.r]) + cur_slice.l
                
            if d1_data[cur_slice.l] >= d1_m and d1_data[cur_slice.r] <= d1_m:
                # print(cur_slice)
                zero_i = zero_bin_search(d1_data[cur_slice.l: cur_slice.r])
                if zero_i != -1:
                    ans.append(zero_i + cur_slice.l)
            
            f_slice = False
            cur_slice.collapse_boarders()
            
        elif not f_slice:
            cur_slice.collapse_boarders()
    
        cur_slice.step()

    if f_slice:
        if d1_data[cur_slice.r] > d1_m:
            cur_slice.r = np.argmin(d1_data[cur_slice.l: cur_slice.r]) + cur_slice.l
            
        if d1_data[cur_slice.l] >= d1_m and d1_data[cur_slice.r] <= d1_m:
            zero_i = zero_bin_search(d1_data[cur_slice.l: cur_slice.r])
            if zero_i != -1:
                ans.append(zero_i + cur_slice.l)
    return np.array(ans)

def combine_groups(groups):
    groups.sort(key=lambda x: len(x), reverse=True)
    d_groups = {}
    for i in range(len(groups)):
        d_groups[i] = True
    res_groups = []
    i = 0
    while i < len(groups):
        j = 0
        while j < len(groups):  # for every group we check all next groups about subsequention
            if i == j:
                j += 1
                continue
            arr1 = groups[i]
            arr2 = groups[j]
            if arr1[0] >= arr2[0]:  # check that arrays are consistent
                j += 1
                continue
            k = check_subsequention(arr1, arr2)  # get len of subsequention
            if k != -1 and k != len(arr2):  # check valide & not full subsequention
                d_groups[i] = False
                d_groups[j] = False
                res_arr = arr1 + arr2[k:]
                if not array_equasion(groups[-1], res_arr):
                    groups.append(res_arr)
                    res_groups.append(res_arr)
            j += 1
        i += 1

    for i in range(len(groups)):
        if d_groups[i]:
            res_groups.append(groups[i])
    return res_groups


def check_subsequention(arr1, arr2) -> int:    
    n = min(len(arr1), len(arr2))
    # print(arr1, arr2, n)
    for i in range(1, n + 1):
        # print(i, arr1[-i:], arr2[:i])
        if arr1[-i] < arr2[0]:
            return -1
        if array_equasion(arr1[-i:], arr2[:i]):
            return i
    return -1


def array_equasion(arr1, arr2) -> bool:
    if len(arr1) != len(arr2):
        return False
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    return True


def in_array(arr1, arr2) -> bool:
    if len(arr1) > len(arr2):
        return False
    for i in range(len(arr2) - len(arr1) + 1):
        if array_equasion(arr1, arr2[i:i + len(arr1)]):
            return True
    return False


def get_valide_groups_from_struct_by_delta(groups_struct, delta, DELTA_DELTA, MIN_GROUP, DELTA_MAX):  # groups_struct = {Point: {Delta: [mistake: float, group: list]}}
    res_groups = []
    if delta < DELTA_MAX:
        # get groups from delta
        for item in groups_struct[delta]:
            if len(item[1]) > MIN_GROUP:
                res_groups.append(item[1])
                
        # get groups from delta +- delta_delta 
        for d2 in groups_struct.keys():
            if d != d2 and d2 < DELTA_MAX and abs(d - d2) / d <= DELTA_DELTA:
                for item in groups_struct[d2]:
                    if len(item[1]) > MIN_GROUP:
                        res_groups.append(item[1])
    return res_groups


def get_valide_groups_from_struct_by_amplitude(groups_struct, d_alpha, peaks):
    res_groups = []
    g_set = set()
    for delta in groups_struct.keys():
        for item in groups_struct[delta]:
            p_arr = item[1]
            n = len(p_arr)
            cur_group = [p_arr[0]]
            group_m_amplitude = d_alpha[peaks[cur_group]][0]
            
            if n <= 2 and p_arr[1] != p_arr[0] + 1:  # check valid by consistance on 0 elemnt
                continue
            
            for i in range(1, n):
                if i < n - 1 and p_arr[i + 1] != p_arr[i] + 1:  # check valid by consistance on i elemnt
                    if abs(d_alpha[peaks[p_arr[i]]] - group_m_amplitude) / max(group_m_amplitude, d_alpha[peaks[p_arr[i]]]) <= 0.34:
                        cur_group.append(p_arr[i])
                        group_m_amplitude = np.nanmean(d_alpha[peaks[cur_group]])
                    break
                
                if abs(d_alpha[peaks[p_arr[i]]] - group_m_amplitude) / max(group_m_amplitude, d_alpha[peaks[p_arr[i]]]) <= 0.34:
                    cur_group.append(p_arr[i])
                    group_m_amplitude = np.nanmean(d_alpha[peaks[cur_group]])
                elif d_alpha[peaks[p_arr[i]]] > group_m_amplitude:
                    if len(cur_group) > 1:
                        res_groups.append(cur_group)
                    
                    cur_group = [p_arr[i]]
                    group_m_amplitude = d_alpha[peaks[cur_group]][0]
            if len(cur_group) > 0:
                group_str = "/".join(list(map(str, cur_group)))
                if group_str not in g_set:
                    g_set.add(group_str)
                    res_groups.append(cur_group)
                
    return res_groups


def get_unique_point_from_groups(groups):
    set_p = set()
    for group in groups:
        set_p = set_p.union(set(group))
    return sorted(list(set_p))


def get_groups_2(arr): # -> groups = {Delta: [mistake: float, group: list]}
    n = len(arr)
    res_struct = {}
    if n < 2:
        return res_struct
    cur_delta = arr[1] - arr[0]
    cur_group = [0, 1]
    cur_err = 0.0
    for i in range(2, n):
        err = abs(arr[i] - (arr[cur_group[-1]] + cur_delta)) - int(random.random() * 10)
        if cur_err + err < cur_delta:  # check valide
            cur_err += err
            cur_group.append(i)
        else:  # save cur group & upd pointer
            if cur_delta in res_struct.keys():
                res_struct[cur_delta].append([cur_err, cur_group])
                res_struct[cur_delta].sort(key=lambda x: len(x[1]), reverse=True)
            else:
                res_struct[cur_delta] = [[cur_err, cur_group]]
            
            cur_delta = arr[i] - arr[cur_group[-1]]
            cur_group = [cur_group[-1], i]
            cur_err = 0.0
    
    if cur_delta in res_struct.keys():
        res_struct[cur_delta].append([cur_err, cur_group])
        res_struct[cur_delta].sort(key=lambda x: len(x[1]), reverse=True)
    else:
        res_struct[cur_delta] = [[cur_err, cur_group]]
    return res_struct


def get_groups(arr): # -> groups = {Delta: [mistake: float, group: list]}
    n = len(arr)
    res_struct = {}
    if n < 2:
        return res_struct
    cur_group = []

    stac_p =  list(range(n - 2, -1, -1))
    while len(stac_p) > 0:
        # print(stac_p, cur_group)
        ind = stac_p.pop()
        # print(ind)
        fl_save = False
        if len(cur_group) == 0:
            cur_delta = arr[ind + 1] - arr[ind]
            cur_group = [ind, ind + 1]
            cur_err = 0.0

            if ind < n - 2:
                stac_p.append(ind + 2)
            else:
                fl_save = True
        else:
            while ind < n - 2 and abs(arr[ind] - (arr[cur_group[-1]] + cur_delta)) >= abs(arr[ind + 1] - (arr[cur_group[-1]] + cur_delta)):
                if ind not in stac_p:
                    stac_p.append(ind)
                ind += 1

            err = abs(arr[ind] - (arr[cur_group[-1]] + cur_delta))
            if cur_err + err < cur_delta:  # check valide
                cur_err += err
                cur_group.append(ind)
                if ind < n - 1:
                    stac_p.append(ind + 1)
                else:
                    fl_save = True
            else:  # save cur group & upd pointer
                fl_save = True
                stac_p.append(ind - 1)
    
        if fl_save or ind == n - 1:
            if cur_delta in res_struct.keys():
                res_struct[cur_delta].append([cur_err, cur_group])
                res_struct[cur_delta].sort(key=lambda x: len(x[1]), reverse=True)
            else:
                res_struct[cur_delta] = [[cur_err, cur_group]]
            cur_group = []
        
    return res_struct


def get_time_delta(arr):
    return (arr[-1] - arr[0]) / (len(arr) - 1)


def merge_peaks(points, d_alpha):
    d = get_time_delta(points)
    res = [points[0]]
    for i in range(1, len(points)):
        if (points[i] - res[-1]) / d < 0.7:
            res[-1] = points[i] if d_alpha[points[i]] > d_alpha[points[i - 1]] else points[i - 1]
        else:
            res.append(points[i])
    return res

def get_groups_from_signal(d_alpha, d_alpha_f, d_alpha_d2f, l_edge, r_edge):  # -> groups = [group: list(time points)]
    DELTA_DELTA = 0.1 # 
    DELTA_MAX = 1000 # points
    MIN_GROUP = 2 # min num points in group

    # get peaks on diagnostic (in the one group)
    peaks = np.array(get_d1_crosses(d_alpha_f, d_alpha_d2f, l_edge, r_edge))

    # # divide peaks to groups
    prev_peaks = []
    cur_peaks = copy.copy(peaks)
    while not array_equasion(prev_peaks, cur_peaks):
        # print("- logg: ", cur_peaks)
        # save peaks time points
        prev_peaks = copy.copy(cur_peaks)
        # get struct of groups = {Delta: [mistake: float, group: list]}
        all_groups_struct = get_groups(cur_peaks)
        # print("- logg: ", all_groups_struct)
        # get list of valid groups by amplitude
        valid_groups = get_valide_groups_from_struct_by_amplitude(all_groups_struct, d_alpha, cur_peaks)
        valid_groups = list(map(lambda x: merge_peaks(x, d_alpha) if len(x) > 1 else x, valid_groups))
        # print("- logg: ", valid_groups)
        # get list of unique peaks indeces
        peaks_ind = get_unique_point_from_groups(valid_groups)
        # get new peaks time points
        cur_peaks = copy.copy(cur_peaks[peaks_ind])
    
    all_groups_struct = get_groups(cur_peaks)
    # print("- logg: ", all_groups_struct)
    valid_groups = get_valide_groups_from_struct_by_amplitude(all_groups_struct, d_alpha, cur_peaks)
    # print("- logg: ", sorted(valid_groups, key=lambda x: x[0]))
    
    # # group post processing: union by delta & get missing
    valid_groups = list(map(lambda x: merge_peaks(x, d_alpha) if len(x) > 1 else x, valid_groups))
    valid_groups = list(filter(lambda x: len(x) > 1, sorted(valid_groups, key=lambda x: x[0])))
    res_peaks = [cur_peaks[valid_groups[0]]]
    for gr_indeces in valid_groups[1:]:
        d1, d2 = get_time_delta(cur_peaks[gr_indeces]), get_time_delta(res_peaks[-1])
        # print("- logg: ", res_peaks[-1], gr_indeces, abs(d1 - d2) / d1)
        if (0.3 < abs(d1 - d2) / d1 < 0.5 and d1 >= d2) or (0.3 < abs(d1 - d2) / d1 < 1.2 and d1 < d2):
            # print("- logg: ", res_peaks[-1], gr_indeces, abs(d1 - d2) / d1)
            # res_peaks.append(merge_peaks(cur_peaks[gr_indeces], d_alpha))
            res_peaks.append(cur_peaks[gr_indeces])
            continue
        
        k = check_subsequention(res_peaks[-1], cur_peaks[gr_indeces])  # get len of subsequention
        if k != -1 and k != len(cur_peaks[gr_indeces]):  # check valide & not full subsequention
            res_arr = np.concatenate([res_peaks[-1], cur_peaks[gr_indeces][k:]])
            # print("- logg: ", res_peaks[-1], cur_peaks[gr_indeces], res_arr)
            # res_peaks[-1] = copy.copy(merge_peaks(res_arr, d_alpha))
            res_peaks[-1] = copy.copy(merge_peaks(res_arr, d_alpha))
        elif in_array(res_peaks[-1], cur_peaks[gr_indeces]):
            # print("- logg: ", res_peaks[-1], cur_peaks[gr_indeces])
            # res_peaks[-1] = copy.copy(merge_peaks(cur_peaks[gr_indeces], d_alpha))
            res_peaks[-1] = copy.copy(cur_peaks[gr_indeces])
        elif k != len(cur_peaks[gr_indeces]) and not in_array(cur_peaks[gr_indeces], res_peaks[-1]):
            # print("- logg: ", k, gr_indeces)
            # res_peaks.append(merge_peaks(cur_peaks[gr_indeces], d_alpha))
            res_peaks.append(cur_peaks[gr_indeces])
    return res_peaks

