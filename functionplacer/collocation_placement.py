from collections import defaultdict
import numpy as np

# VM class
class VM:
    def __init__(self, vm_id, cpu, mem, type):
        self.vm_id = vm_id
        self.cpu_capacity = cpu
        self.mem_capacity = mem
        self.available_cpu = cpu
        self.available_mem = mem
        self.placements = []
        self.type = type

    def can_host(self, cpu, mem):
        return self.available_cpu >= cpu and self.available_mem >= mem

    def place(self, func_id, cpu, mem):
        self.available_cpu -= cpu
        self.available_mem -= mem
        self.placements.append((func_id, cpu, mem))

    def resource_ratio(self):
        return self.cpu_capacity / self.mem_capacity


class Function:
    def __init__(self, func_id, cpu, mem, count):
        self.func_id = func_id
        self.cpu = cpu
        self.mem = mem
        self.count = count  # Number of containers to place

    def resource_ratio(self):
        return self.cpu / self.mem


def function_collocation_placement(vms, functions):
    N = len(functions)
    vm_pool = sorted(vms, key=lambda v: v.resource_ratio())
    M = len(vm_pool)

    # Step 1: Build collocation matrix
    collocation_matrix = []
    for i in range(N):
        for j in range(i + 1, N):
            f1, f2 = functions[i], functions[j]
            hij = min(f1.count, f2.count)
            aij = f1.cpu + f2.cpu
            bij = f1.mem + f2.mem
            rij = aij / bij
            collocation_matrix.append({
                'pair': (f1.func_id, f2.func_id),
                'cpu': aij,
                'mem': bij,
                'ratio': rij,
                'count': hij
            })

    def cal_dominant_ratio():
        total_weight = sum(e['count'] for e in collocation_matrix)
        if total_weight == 0:
            dominant_ratio = 0
        else:
            dominant_ratio = sum((e['ratio'] * e['count']) for e in collocation_matrix) / total_weight
        return dominant_ratio

    # Step 3: Place collocated pairs greedily
    placements = defaultdict(list)
    dominant_ratio = cal_dominant_ratio()
    for m in range(M):
        vm = sorted(vm_pool, key=lambda x: abs(dominant_ratio - vm.resource_ratio()))[0]
        sorted_pairs = sorted(collocation_matrix, key=lambda x: abs(x['ratio'] - vm.resource_ratio()))
        for entry in sorted_pairs:
            f1_id, f2_id = entry['pair']
            count = entry['count']
            if count <= 0:
                continue

            f1 = next(f for f in functions if f.func_id == f1_id)
            f2 = next(f for f in functions if f.func_id == f2_id)

            while (f1.count > 0 and f2.count > 0 and
                   vm.can_host(f1.cpu + f2.cpu, f1.mem + f2.mem)):
                vm.place(f1_id, f1.cpu, f1.mem)
                vm.place(f2_id, f2.cpu, f2.mem)
                placements[vm.vm_id].append((f1_id, f2_id))
                f1.count -= 1
                f2.count -= 1
                entry['count'] -= 1

    # Step 4: First-Fit for remaining containers
    for f in functions:
        while f.count > 0:
            placed = False
            for vm in vm_pool:
                if vm.can_host(f.cpu, f.mem):
                    vm.place(f.func_id, f.cpu, f.mem)
                    placements[vm.vm_id].append((f.func_id,))
                    f.count -= 1
                    placed = True
                    break
            if not placed:
                break  # No more placement possible

    return placements, vm_pool
