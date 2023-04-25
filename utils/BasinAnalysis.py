import numpy as np
from matplotlib import pyplot as plt


from skimage.measure import block_reduce

class Analysis:

    def __init__(self, ls, vars, diff_vars, data_args_list):
        """
        ls: list of level exponenents (see `BasinInit.py`)
        vars_file: path to npz as result of `run_LvlVar.py`. ATTENTION: with same ls as given here
        diff_vars_file: path to npz as result of `run_LvlVar.py`. ATTENTION: with same ls as given here
        """
        self.ls = ls
        
        self.vars = vars
        if isinstance(vars, str):
            self.vars = np.load(vars)
        
        self.diff_vars = diff_vars
        if isinstance(diff_vars, str):
            self.diff_vars = np.load(diff_vars)

        if len(self.diff_vars) == len(self.ls):
            self.diff_vars = self.diff_vars[1:]

        assert len(self.ls) == len(self.vars), "Wrong number of levels"
        assert len(self.ls) == len(self.diff_vars) + 1, "Wrong number of levels"

        self.dxs = [data_args["dx"] for data_args in data_args_list]
        self.dys = [data_args["dy"] for data_args in data_args_list]

        self.nxs = [data_args["nx"] for data_args in data_args_list]
        self.nys = [data_args["ny"] for data_args in data_args_list]



    def plotLvlVar(self, relative=False):

        if relative: 
            vars = self.vars/self.vars[-1]
            diff_vars = self.diff_vars/self.vars[-1]
        else:
            vars = self.vars
            diff_vars = self.diff_vars

        with plt.rc_context({'lines.color':'black', 
                        'text.color':'black', 
                        'axes.labelcolor':'black', 
                        'xtick.color':'black',
                        'ytick.color':'black'}):
            fig, axs = plt.subplots(1,3, figsize=(15,5))

            Nxs = [nx*ny for nx, ny in zip(self.nxs, self.nys)]
            for i in range(3):
                axs[i].loglog(Nxs, vars[:,i], label="$|| Var[u^l] ||_{L^2}$", linewidth=3)
                axs[i].loglog(Nxs[1:], diff_vars[:,i], label="$|| Var[u^l-u^{l-1}] ||_{L^2}$", linewidth=3)
                axs[i].set_xlabel("# grid cells", fontsize=15)
                # axs[i].set_ylabel("variance", fontsize=15)
                axs[i].legend(labelcolor="black", loc=(0.2,0.5), fontsize=15)

                axs[i].set_xticks(Nxs)
                axs[i].xaxis.grid(True)
                axs[i].set_xticklabels(["$64$","$128$","$256$","$512$","$1024$"])

                # for l_idx, l in enumerate(ls):
                #     axs[i].annotate(str(l), (Nxs[l_idx], 1e-5), color="black")

            axs[0].set_title("$\eta$", fontsize=15)
            axs[1].set_title("$hu$", fontsize=15)
            axs[2].set_title("$hv$", fontsize=15)



    def _level_work(self, l):
        """
        Cubic work in terms of grid discretisation

        The dx should be in synv with `BasinInit.py`
        """
        dx = 1/2*(self.dxs[-1] + self.dys[-1])
        return dx**(-3)


    def optimal_Ne(self, tau):
        """
        Evaluating the optimal ML ensemble size for a error level `tau`

        See Ch. 5 of Kjetils thesis for reference 
        """

        rel_vars = self.vars/self.vars[-1]
        rel_diff_vars = self.diff_vars/self.vars[-1]

        avg_vars = np.mean(rel_vars, axis=1)
        avg_diff_vars = np.mean(rel_diff_vars, axis=-1)


        allwork = 0
        for l_idx, l in enumerate(self.ls):
            if l_idx == 0: 
                allwork += np.sqrt(avg_vars[l_idx] * self._level_work(l))
            else:
                allwork += np.sqrt(avg_diff_vars[l_idx-1] * self._level_work(l))

        optNe_ref = np.zeros(len(self.ls))
        for l_idx, l in enumerate(self.ls):
            if l_idx == 0: 
                optNe_ref[l_idx] = np.sqrt(avg_vars[l_idx]/((2**l)**2)) * allwork
            else: 
                optNe_ref[l_idx] = np.sqrt(avg_diff_vars[l_idx-1]/((2**l)**2)) * allwork

        return np.int32(np.ceil(1/(tau**2)*optNe_ref))

    
    def work(self, Nes):
        """
        Evaluating the theoretical error for an ML ensemble
        work(0 and + ensemble members) + work(- ensemble members)
        """
        assert len(Nes) == len(self.ls), "Wrong number of levels"
        return np.sum([self._level_work(l) for l in self.ls] * np.array(Nes)) + np.sum([self._level_work(l) for l in self.ls[:-1]] * np.array(Nes[1:]))

    
