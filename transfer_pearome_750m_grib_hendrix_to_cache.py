import vortex
from vortex import toolbox
from vortex import tools
import numpy as np
import os
from bronx.stdtypes.date import daterangex,timeintrangex,timerangex
import common
cutoff="p"
vconf="pefrance"
suite="H3Z3"
geometry="PARIS1S100"
print(toolbox.defaults.show())

os.environ['MTOOL_STEP_CACHE']="/scratch/mtool/selvarajd/cache"
dico={
"experiment":suite,
"date":daterangex("2024090821","2024091121","PT24H"),
"model":'arome',
"origin":"hst",
"kind":'gridpoint',
"block":'forecast',
"cutoff":cutoff,
"vapp":'arome',
"vconf":vconf,
"geometry":geometry,
"namespace":'vortex.multi.fr',
"nativefmt":'grib',
"member":list(range(0,25)),
"term":list(range(1,51)),
"shouldfly":True
}

def get_hendrix(rh_i):
	return rh_i.get()

ftp_v = np.vectorize(get_hendrix)

rhlist=toolbox.rload(**dico)
print(rhlist)
t = vortex.ticket()
sh = t.sh
sh.prompt = t.prompt
sh.ftraw = True
with t.sh.ftppool():
    ftp_v(rhlist) #for rh in rhlist:
    #    rh.get()
    #    ##--------REMOVING THE shouldfly-* files of vortex from the working directory----
    cmd_should = 'rm -rf shouldfly-*'
    os.system(cmd_should)
	
