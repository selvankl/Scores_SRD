import vortex
from vortex import toolbox
from vortex import tools
import numpy as np
import os
from bronx.stdtypes.date import daterangex,timeintrangex,timerangex
import common
from concurrent.futures import ProcessPoolExecutor as pool
#:meth:`~OSExtended.ftserv_batchget`


cutoff="p"
vconf="pefrance"
suite="DBLE"
geometry="EURW1S40"
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
"term":list(range(0,52)),
"shouldfly":True
}

#"member":list(range(0,25)),

def get_hendrix(rh_i):
    rh_i.get()
    cmd_should = 'rm -rf shouldfly-*'
    os.system(cmd_should)
    return None

ftp_v = np.vectorize(get_hendrix)

rhlist=toolbox.rload(**dico)
#print(rhlist)
t = vortex.ticket()
sh = t.sh
sh.prompt = t.prompt
sh.ftraw = True


if __name__ == "__main__":
    
    #with pool(max_workers=10) as P:
    #    ftp_pool_dwnl = P.map(get_hendrix, rhlist)
    with t.sh.ftppool():
        ftp_v(rhlist)

    #ftserv_batchget(rhlist)
    #ftp_v(rhlist) #for rh in rhlist:
    #    rh.get()
    #    ##--------REMOVING THE shouldfly-* files of vortex from the working directory----
    #cmd_should = 'rm -rf shouldfly-*'
    #os.system(cmd_should)
	
