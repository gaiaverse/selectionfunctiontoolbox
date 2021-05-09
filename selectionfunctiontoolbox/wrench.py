#!/usr/bin/env python
#
# Models a selection function that is independent between positions.
#
# Copyright (C) 2021  Douglas Boubert & Andrew Everall
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

import numpy as np
import healpy as hp
import tqdm
import h5py
from .base import Base
import os

class Wrench(Base):

    basis_keyword = 'basic'

    def _process_basis_options(self):
        self.S = self.P
        self.spherical_basis_file = f'{self.basis_keyword}.h5'

    def _generate_spherical_basis(self,gsb_file):
        with h5py.File(gsb_file, 'w') as f:
            pass