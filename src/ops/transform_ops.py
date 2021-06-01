import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog('rdApp.*')


def mol_from_string(string: str, catch_errors: bool = False) -> Chem.Mol:
    '''
    In the very most cases, the following would be enough:

        def mol_from_string(smiles):
            if string.startswith('InChI'):
                mol = Chem.MolFromInchi(string, sanitize=True)
            else:
                mol = Chem.MolFromSmiles(string, sanitize=True)
            return mol

    However, because sanitization errors could occur during
    the sanitization procedure, it may be desirable to catch
    this error and redo the sanitization without the parameter
    causing the error.
    '''

    if string.startswith('InChI'):
        def transform_fn(x, sanitize=True):
            try:
                mol = Chem.MolFromSmiles(
                    Chem.MolToSmiles(
                        Chem.MolFromInchi(
                            x, sanitize=sanitize
                        )
                    ), sanitize=sanitize
                )
                return mol
            except:
                return None
    else:
        transform_fn = Chem.MolFromSmiles

    if not catch_errors:
        return transform_fn(string, sanitize=True)

    mol = transform_fn(string, sanitize=False)
    flag = Chem.SanitizeMol(mol, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        mol = transform_fn(string, sanitize=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^flag)
    return mol


def ecfp_from_string(string, bits, radius, use_counts):

    def vectorize(fp, bits, use_counts):
        vec = np.zeros(bits, dtype='int32')
        vec[list(fp.keys())] = (
            np.array(list(fp.values())) if use_counts else 1
        )
        return vec

    mol = mol_from_string(string, False)
    if mol is None: return np.zeros(bits)

    ecfp = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=bits).GetCountFingerprint(mol).GetNonzeroElements()
    return vectorize(ecfp, bits, use_counts)

def desc_from_string(string, thresh=None):

    mol = mol_from_string(string, False)
    if mol is None: return [0] * len(Descriptors.descList)

    mol_descriptors = []
    for name, func in Descriptors.descList:
        try:
            # descriptor qed sometimes throws an error as follows:
            # AtomValenceException: Explicit valence for atom # 14 N, 4, is greater than permitted
            # for the same molecule, these descriptors will return nan:
            #   BCUT2D_MWHI
            #   BCUT2D_MWLOW
            #   BCUT2D_CHGHI
            #   BCUT2D_CHGLO
            #   BCUT2D_LOGPHI
            #   BCUT2D_LOGPLOW
            #   BCUT2D_MRHI
            #   BCUT2D_MRLOW
            if name == 'Ipc':
                # if avg=True is not passed, Ipc will take extremely high values,
                # causing overflow when e.g. RF converts values to float32
                descriptor = func(mol, avg=True)
            else:
                descriptor = func(mol)

            if np.isnan(descriptor):
                # no error was thrown, but NaN was returned
                raise Exception('NaN was returned')
        except Exception as e:
            # print(f"Desc '{name}': {e}") # uncomment for debugging
            descriptor = 0.0 # set to 0 as descriptor could not be calculated

        mol_descriptors.append(descriptor)

    return mol_descriptors
