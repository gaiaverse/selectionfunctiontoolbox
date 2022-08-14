import sys, os
import pandas as pd, h5py, numpy as np
from astroquery.gaia import Gaia

if __name__=='__main__':

    print(sys.argv)
    output_dir, data_file, Gres, Cres, healpix = sys.argv[1:]
    Gres=float(Gres); Cres=float(Cres)

    Gmin=1.6

    query = f'''
    SELECT magnitude, colour, position, count(*) as n, sum(selection) as k
                     FROM ( SELECT  to_integer(floor((phot_g_mean_mag)/{Gres})) as magnitude,
                                    to_integer(floor((phot_g_mean_mag-phot_rp_mean_mag)/{Cres})) as colour,
                                    gaia_healpix_index({healpix}, source_id) as position,
                                    to_integer(IF_THEN_ELSE(rv_nb_transits>0 , 1.0, 0.0)) as selection
                            FROM gaiadr3.gaia_source where phot_g_mean_mag between {Gmin} and 17.6
                                                        and phot_rp_mean_mag is not null
                                                        and random_index<10000000000) as subquery
                     GROUP BY magnitude, colour, position
    '''
    print(query)

    job = Gaia.launch_job_async(query)
    data_df = job.get_results().to_pandas()

    with h5py.File(os.path.join(output_dir, data_file), 'w') as hf:
        for key in data_df.keys():
            hf.create_dataset(key, data=data_df[key].astype(np.int32))
