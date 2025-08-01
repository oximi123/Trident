import random

import gym
import pandas as pd
import numpy as np

from functionplacer.collocation_placement import Instance

path_prefix = ''


class VM_type:
    def __init__(self, config, empty_num):
        self.config = config
        self.cpu = config[0]
        self.mem = config[1]
        self.empty_num = empty_num
        self.is_collocated = False
        self.instances = [Instance(id, self.cpu, self.mem) for i in range(self.empty_num)]

    def clear(self):
        for instance in self.instances:
            instance.clear()
            instance.clear()

    def best_fit_place(self, function_cpu, function_mem):
        best_instance_id = -1
        best_mem = 1000000
        for instance_id, instance in enumerate(self.instances):
            idle_cpu, idle_mem = instance.available_cpu, instance.available_mem
            if idle_cpu >= function_cpu and idle_mem >= function_mem:
                if best_mem > idle_mem:
                    best_mem = idle_mem
                    best_instance_id = instance_id
        return best_instance_id, best_mem

    def first_fit_place(self, function_cpu, function_mem):
        for instance_id, instance in enumerate(self.instances):
            idle_cpu, idle_mem = instance.available_cpu, instance.available_mem
            if idle_cpu >= function_cpu and idle_mem >= function_mem:
                return instance_id
        return -1

    def collocate(self, function_cpu, function_mem, max_num):
        collocate_num = 0
        for instance_id, instance in enumerate(self.instances):
            n = int(min((instance.available_cpu) / function_cpu, (instance.available_mem) / function_mem))
            if collocate_num + n <= max_num:
                self.place(n * function_cpu, n * function_mem, instance_id)
            else:
                n = max_num - collocate_num
                self.place(n * function_cpu, n * function_mem, instance_id)
                collocate_num = min(collocate_num + n, max_num)
                break
            collocate_num = min(collocate_num + n, max_num)

        return collocate_num

    def place(self, function_cpu, function_mem, instance_id):
        self.instances[instance_id][0] += function_cpu
        self.instances[instance_id][1] += function_mem

    def resource_usage_rate(self):
        mem_usages = []
        cpu_usages = []
        for instance in self.instances:
            if self.cpu - instance.available_cpu != 0:
                cpu_usages.append(self.cpu - instance.available_cpu / self.config[0])
                mem_usages.append(self.mem - instance.available_mem / self.config[1])
        return cpu_usages, mem_usages

    def resource_usage(self):
        cpu_used = 0
        mem_used = 0
        cpu_allocated = self.empty_num * self.cpu
        mem_allocated = self.empty_num * self.mem
        for instance in self.instances:
            cpu_used += self.cpu - instance.available_cpu
            mem_used += self.mem - instance.available_mem
        return cpu_used, mem_used, cpu_allocated, mem_allocated


def load_vm2config():
    vm_config = pd.read_csv(path_prefix + 'data/vmConfig.csv')
    vm2config = {}
    group2vms = {}
    for index, row in vm_config.iterrows():
        vm2config[row[0]] = {
            'cpu': row[1],
            'mem': row[2]
        }
    return vm2config


def load_vm2fail():
    vm_types = set()
    for vm in vm2config.keys():
        if vm.startswith("Spot"):
            vm_types.add(vm.split('-')[1])
    failprob = pd.read_csv(path_prefix + 'data/failureProbaility.csv', usecols=vm_types)
    vm2fail = {}
    for column in failprob.columns:
        vm2fail['Spot-' + column] = failprob[column].tolist()
    for vm in vm2config.keys():
        if vm not in vm2fail:
            vm2fail[vm] = [0 for _ in range(1440)]
    return vm2fail


def load_vm2price(ondemand_scale=4):
    spot_vm_types = set()
    ondemand_vm_types = set()
    for vm in vm2config.keys():
        if vm.startswith("Spot"):
            spot_vm_types.add(vm.split('-')[1])
        else:
            ondemand_vm_types.add(vm.split('-')[1])
    spot_price = pd.read_csv(path_prefix + 'data/FullCurrentPrice.csv', usecols=spot_vm_types)
    vm2price = {}
    for column in spot_price.columns:
        price = spot_price[column].tolist()
        idx = 0
        for id, i in enumerate(price):
            if i > 0:
                idx = id
                break
        avg = sum(price[idx:]) / (len(price) - idx)
        price[0:idx] = [avg for _ in range(idx)]
        vm2price['Spot-' + column] = np.array(price)
        vm2price['Ondemand-' + column] = np.array(price) * ondemand_scale
    return vm2price


def load_workload():
    workload = pd.read_csv(path_prefix + 'data/workload.csv')
    mem_per_request = workload['mem_per_request'].to_numpy()
    cpu_per_request = workload['cpu_per_request'].to_numpy()
    execution_time = workload['execution_time'].to_numpy()
    request_pattern = workload.drop(['HashApp', 'HashFunction', 'mem_per_request', 'execution_time', 'cpu_per_request'],
                                    axis=1).to_numpy()
    return mem_per_request, request_pattern, cpu_per_request, execution_time


vm2config = load_vm2config()
vm2fail = load_vm2fail()
vm2price = load_vm2price()
mem_per_request, request_pattern, cpu_per_request, execution_time = load_workload()

class ServerlessEnv(gym.Env):

    def __init__(self, prediction_model, placement_algorithm, K=6, SLO=0.9, horizon_length=24, period_length=20, data_len = 1440,
                 forward_steps=10, alpha=1e-3,
                 beta=100, iter_steps=10, init_method='weighted', action_step=0.01, random_trace=False, train=True):
        super(ServerlessEnv, self).__init__()
        self.prediction_model = prediction_model
        self.placement_algorithm = placement_algorithm
        self.num_agents = K
        self.horizon_length = horizon_length
        self.period_length = period_length
        self.train = train
        self.action_step = action_step
        self.low_level_action_space = 20
        self.high_level_action_space = len(vm2config)
        self.SLO = SLO
        self.provisioned_VM = {}
        self.current_time = 0
        self.current_period = 0
        self.data_len = data_len
        self.alpha = alpha
        self.beta = beta
        self.forward_steps = forward_steps
        self.iter_steps = iter_steps
        self.low_level_steps_left = iter_steps
        self.provisioned_unit = 1
        self.id2vm = {}
        self.init_cost = 0
        self.current_utility = 0
        self.current_vms = []
        self.current_vm_weight = []
        self.init_method = init_method
        self.action_space = len(vm2config)
        self.trace_start = 0
        self.random_trace = random_trace
        self.high_level_observation_space = 10 * 2 + len(vm2config)
        self.low_level_observation_space = len(vm2config) + 3 + 40
        for id, vm in enumerate(vm2config.keys()):
            self.id2vm[id] = vm
        self.action2provision = {
            0: 0,
            1: 1,
            2: -1
        }
        self.reward = {
            'high-level': None,
            'low-level': None
        }
        self.is_done = {
            'high-level': False,
            'low-level': [False for _ in range(self.num_agents)]
        }
        self.set_mem = 0
        self.low_level_best_provision = {}
        self.low_level_best_util = -100000000000000000
        self.__init__data()

    def get_agent_num(self):
        return self.num_agents

    def __init__data(self):
        self.cpu_per_request, self.mem_per_request, self.request_pattern, self.execution_time = (
            cpu_per_request,
            mem_per_request,
            request_pattern[:, self.trace_start:self.trace_start + self.data_len],
            execution_time)

    def get_cur_failure(self, vm):
        return np.average(vm2fail[vm][
                          self.current_period * self.forward_steps: self.current_period * self.forward_steps + self.forward_steps])

    def __init_vm_purchase__(self):
        init_provisioned_VM = {}
        for vm in self.current_vms:
            init_provisioned_VM[vm] = 0
        init_cost = 0
        request_mem = self.request_pattern[:, self.current_period * self.forward_steps
                                              : (self.current_period + 1) * self.forward_steps] * np.array(
            [self.mem_per_request for _ in range(self.forward_steps)]).T
        request_mem = max(request_mem.sum(axis=0)) * self.SLO

        request_cpu = self.request_pattern[:, self.current_period * self.forward_steps
                                              : (self.current_period + 1) * self.forward_steps] * np.array(
            [self.cpu_per_request for _ in range(self.forward_steps)]).T
        request_cpu = max(request_cpu.sum(axis=0)) * self.SLO
        if self.init_method == 'avg':
            for vm in self.provisioned_VM.keys():
                self.provisioned_VM[vm] = 0
            current_mem = 0
            current_cpu = 0
            mem_tmp = 0
            cpu_tmp = 0
            for vm in self.current_vms:
                mem_tmp += vm2config[vm]['mem']
                cpu_tmp += vm2config[vm]['cpu']
            while current_mem < request_mem or current_cpu < request_cpu:
                for vm in self.current_vms:
                    init_provisioned_VM[vm] += 1
                    init_cost += sum(vm2price[vm][self.current_period * self.forward_steps
                                                  : (self.current_period + 1) * self.forward_steps])
                current_mem += mem_tmp
                current_cpu += cpu_tmp
        elif self.init_method == 'weighted':
            for id, vm in enumerate(self.current_vms):
                try:
                    init_provisioned_VM[vm] = int(max(request_cpu * self.current_vm_weight[id] / vm2config[vm]['cpu'],
                                                      request_mem * self.current_vm_weight[id] / vm2config[vm]['mem']))
                except ValueError:
                    print(self.current_vm_weight)
                    print(vm2config)
                if vm.startswith('Spot'):
                    init_provisioned_VM[vm] = int(max(request_cpu * self.current_vm_weight[id] / vm2config[vm]['cpu'],
                                                      request_mem * self.current_vm_weight[id] / vm2config[vm][
                                                          'mem']) * 1.2)
                init_cost += sum(vm2price[vm][self.current_period * self.forward_steps
                                              : (self.current_period + 1) * self.forward_steps]) * init_provisioned_VM[
                                 vm]
        self.init_cost = init_cost
        return init_provisioned_VM

    def action2vm(self, action):
        idx = (np.argsort(action)[::-1])[0:self.num_agents]
        selected_vm = []
        vms = list(vm2config.keys())
        for i in idx:
            selected_vm.append(vms[i])
        if sum(action[idx]) == 0:
            return selected_vm, np.array([1.0 / self.num_agents] * self.num_agents)
        return selected_vm, np.array(action[idx]) / sum(action[idx])

    def high_level_step(self, high_level_action):
        self.current_vms, self.current_vm_weight = self.action2vm(high_level_action)
        self.id2vm = {}
        for id, vm in enumerate(self.current_vms):
            self.id2vm[id] = vm
        self.provisioned_VM = self.__init_vm_purchase__()
        self.low_level_best_provision = self.provisioned_VM.copy()
        self.current_utility = self.__utility__()
        self.low_level_best_util = self.current_utility
        return self.current_utility

    def low_level_step(self, low_level_actions):
        previous_utility = self.__utility__()
        for id, act in enumerate(low_level_actions):
            self.provisioned_VM[self.id2vm[id]] = int(
                max(0, self.provisioned_VM[self.id2vm[id]] * (1 + self.action2provision[
                    act] * self.action_step)))
        current_utility = self.__utility__()
        self.current_utility = current_utility
        if self.low_level_best_util < current_utility:
            self.low_level_best_util = current_utility
            self.low_level_best_provision = self.provisioned_VM.copy()
        low_level_reward_t = current_utility - previous_utility
        if low_level_reward_t < 0:
            self.low_level_steps_left -= 2
        else:
            self.low_level_steps_left -= 1
        low_level_done = [False for _ in range(self.num_agents)]
        if self.low_level_steps_left <= 0:
            self.provisioned_VM = self.low_level_best_provision
            low_level_done = [True for _ in range(self.num_agents)]
            self.low_level_steps_left = self.iter_steps

        return self.get_low_level_state(), low_level_reward_t, low_level_done, {}

    def step(self, action=None):
        self.current_period += 1
        if self.current_period * self.forward_steps >= self.data_len:
            self.is_done['high-level'] = True
            return self.is_done['high-level']
        else:
            self.is_done['high-level'] = False
            return self.is_done['high-level']

    def low_level_reset(self):
        self.provisioned_VM = {}
        self.low_level_best_util = -10000000
        self.low_level_best_provision = {}

    def reset(self):
        if self.random_trace:
            self.trace_start = self.data_len * (random.randint(0, request_pattern.shape[1] // 2 // self.data_len))
            self.request_pattern = request_pattern[:, self.trace_start:self.trace_start + self.data_len]
        self.current_time = 0
        self.current_period = 0
        self.current_utility = 0
        self.low_level_steps_left = self.iter_steps
        self.reward = {
            'high-level': None,
            'low-level': None
        }
        self.is_done = {
            'high-level': False,
            'low-level': [False for _ in range(self.num_agents)]
        }
        self.provisioned_VM = {}
        self.current_utility = self.__utility__()

    def get_VMtypes(self):
        vm_types = []
        evicted_vm = self.__eviction__(self.provisioned_VM)
        for id, vm in enumerate(self.provisioned_VM.keys()):
            vm_types.append(VM_type([vm2config[vm]['cpu'], vm2config[vm]['mem']], evicted_vm[vm]))

        return vm_types

    def __place__(self, train=True):
        info = {
            'cost': 0,
            'slo': 0,
            'resource_usage': [0, 0]
        }
        vm_types = self.get_VMtypes()
        if not train:
            failed_requests = np.zeros(len(self.request_pattern)).astype(int)
            avg_duration_resource_usage = np.array([0, 0]).astype(float)
            for i in range(self.current_period * self.forward_steps, (self.current_period + 1) * self.forward_steps):
                for vm_type in vm_types:
                    vm_type.clear()
                failed_request, resource_util = self.place_algorithm(self.mem_per_request, self.cpu_per_request,
                                                                     self.request_pattern[:, i].astype(int), vm_types)
                failed_requests = failed_requests + failed_request
                avg_duration_resource_usage += resource_util

            resource_utils = avg_duration_resource_usage / self.forward_steps
            sum_function_requests = self.request_pattern[:,
                                    self.current_period * self.forward_steps: self.current_period * self.forward_steps + self.forward_steps].sum(
                axis=1)
            idx = np.where(sum_function_requests > 0)
            sum_function_requests = sum_function_requests[idx]
            slos = (sum_function_requests - failed_requests[idx]) / (sum_function_requests)
            slo = sum(slos) / len(slos)
            info['slo'] = slo
            info['cost'] = self.get_provision_cost()
            info['resource_usage'] = resource_utils
        else:
            function_requests = np.average(self.request_pattern[:,
                                           self.current_period * self.forward_steps: self.current_period * self.forward_steps + self.forward_steps],
                                           axis=1)
            failed_requests, resource_utils = self.place_algorithm(self.mem_per_request, self.cpu_per_request,
                                                                   function_requests.astype(int), vm_types)
            idx = np.where(function_requests > 0)
            sum_function_requests = function_requests[idx]
            slos = (sum_function_requests - failed_requests[idx]) / (sum_function_requests)
            slo = sum(slos) / len(slos)
        slo_vio = max(0, self.SLO - slo)
        return slo_vio * self.forward_steps, info

    def get_provision_cost(self):
        cost = 0
        for vm in self.provisioned_VM:
            provision_num = self.provisioned_VM[vm]
            vm_price = vm2price[vm][
                       self.current_period * self.forward_steps: self.current_period * self.forward_steps + self.forward_steps]
            cost += sum((provision_num * vm_price))
        return cost

    def __eviction__(self, provisioned_vm):
        provisioned_vm = provisioned_vm.copy()
        for vm in provisioned_vm.keys():
            if vm.startswith('Spot'):
                avg_fail_rate = sum(vm2fail[vm][self.current_period * self.forward_steps: (
                                                                                                  self.current_period + 1) * self.forward_steps]) / self.forward_steps
                provisioned_vm[vm] = int(provisioned_vm[vm] * (1 - avg_fail_rate))
        return provisioned_vm

    def __utility__(self):
        if self.is_done['high-level']:
            return 0
        provisioned_vm_tmp = self.provisioned_VM.copy()
        provisioned_vm_tmp = self.__eviction__(provisioned_vm_tmp)
        provision_cost = self.get_provision_cost()
        SLO_violation, _ = self.__place__(self.train)

        return - provision_cost * self.alpha - SLO_violation * self.beta

    def get_available_resource(self, provisioned_VM):
        mem = 0
        cpu = 0
        for vm in self.provisioned_VM.keys():
            mem += provisioned_VM[vm] * vm2config[vm]['mem']
            cpu += provisioned_VM[vm] * vm2config[vm]['cpu']
        return mem, cpu

    def get_high_level_state(self):
        if self.is_done['high-level']:
            return np.zeros(self.high_level_observation_space)
        else:
            request_pattern_next_period = self.request_pattern[:, self.current_period * self.forward_steps
                                                                  : (self.current_period + 1) * self.forward_steps]
            mem_per_request_next_period = np.array([self.mem_per_request for i in range(self.forward_steps)]).reshape(
                request_pattern_next_period.shape)
            high_level_mem_next_period = (request_pattern_next_period * mem_per_request_next_period).sum(
                axis=0)

            cpu_per_request_next_period = np.array([self.cpu_per_request for i in range(self.forward_steps)]).reshape(
                request_pattern_next_period.shape)
            high_level_cpu_next_period = (request_pattern_next_period * cpu_per_request_next_period).sum(
                axis=0)

            provisioned_vms = list(self.provisioned_VM.values())

            vm_info = []
            for vm in vm2config:
                price = np.average(vm2price[vm][
                                   self.current_period * self.forward_steps: self.current_period * self.forward_steps + self.forward_steps])
                failure = np.average(vm2fail[vm][
                                     self.current_period * self.forward_steps: self.current_period * self.forward_steps + self.forward_steps])
                cpu = vm2config[vm]['cpu']
                mem = vm2config[vm]['mem']
                vm_info.append((cpu * (1 - failure) + mem * (1 - failure) / price) / 100)
            high_level_state = np.concatenate(
                [high_level_mem_next_period, high_level_cpu_next_period,
                 vm_info
                 ])
        return high_level_state

    def get_low_level_state(self):
        evicted_vm = self.__eviction__(self.provisioned_VM.copy())
        available_mem, available_cpu = self.get_available_resource(evicted_vm)

        request_pattern_next_period = self.request_pattern[:, self.current_period * self.forward_steps
                                                              : (self.current_period + 1) * self.forward_steps]
        mem_per_request_next_period = np.array([self.mem_per_request for i in range(self.forward_steps)]).reshape(
            request_pattern_next_period.shape)
        mem_next_period = (request_pattern_next_period * mem_per_request_next_period).sum(
            axis=0)
        mem_residual = (mem_next_period - np.array(
            [available_mem for _ in range(len(mem_next_period))]))

        cpu_per_request_next_period = np.array([self.cpu_per_request for i in range(self.forward_steps)]).reshape(
            request_pattern_next_period.shape)
        cpu_next_period = (request_pattern_next_period * cpu_per_request_next_period).sum(
            axis=0)
        cpu_residual = (mem_next_period - np.array(
            [available_cpu for _ in range(len(cpu_next_period))]))

        if self.is_done['low-level'][0]:
            return [np.zeros(self.low_level_observation_space.shape) for _ in range(self.num_agents)]
        else:
            low_level_state = []
            vm_nums = np.zeros(len(vm2config))
            for i, vm in enumerate(vm2config.keys()):
                if vm in self.provisioned_VM.keys():
                    vm_nums[i] = self.provisioned_VM[vm]
            for vm in self.current_vms:
                vm_config = [vm2config[vm]['mem'], vm2config[vm]['cpu']]
                vm_fail = vm2fail[vm][
                          self.current_period * self.forward_steps: (self.current_period + 1) * self.forward_steps] if \
                    not self.is_done['low-level'][0] else np.zeros((self.forward_steps))
                vm_price = vm2price[vm][
                           self.current_period * self.forward_steps: (self.current_period + 1) * self.forward_steps] if \
                    not self.is_done['low-level'][0] else np.zeros((self.forward_steps))
                low_level_state.append(np.concatenate(
                    [mem_residual, cpu_residual, vm_nums, vm_config,
                     vm_fail, vm_price, [self.current_utility]]))
        return low_level_state

    def get_high_level_reward(self):
        return self.__utility__()

    def on_demand_cost(self, request_cpu, request_mem):
        lowest_cost = 100000000000
        best_vm = None
        for vm in vm2config.keys():
            if vm.startswith('Spot'):
                continue
            num = max(request_cpu / vm2config[vm]['cpu'], request_mem / vm2config[vm]['mem'])
            cost = sum(vm2price[vm][
                       self.current_period * self.forward_steps: self.current_period * self.forward_steps + self.forward_steps]) * num
            if cost < lowest_cost:
                best_vm = vm
            lowest_cost = min(lowest_cost, cost)
        return lowest_cost
