from typing import Dict, Union, List
import os
import yaml
import numpy as np
from scipy import sparse
from PIL import Image


class PopGenDataClass:
    """Defines base data class for genetic data handling."""

    def __init__(self,
                 config: Dict = None,
                 verbose_level: int=0,
                 init_data_dir: bool=True,
                 init_log: bool=True):
        """Initializes popgen dataclass with the specified configuration

        Parameters
        ----------
        config: (Dict) - Dictionary containing the configuration for the data class.
        verbose_level: (int) - Controls level of verbosity for the class
        init_data_dir: (bool) - If True, initializes the data directorys
        """

        # Setup print statements
        if init_log:
            global log
            if verbose_level > 0:
                def log(x, level: int = 1):
                    if level <= verbose_level:
                        print(x)
            else:
                def log(x, level: int = 1):
                    pass
            self.log = log

        # Set config
        if config is not None:
            self.config = config

            # Set directories for saving
            self.base_dir = os.path.join(os.getcwd(), self._root_dir_name(), self._data_tree())
            if not os.path.isdir(self.base_dir):
                os.makedirs(self.base_dir)
            if init_data_dir:
                self.data_dir = self._find_or_make_data_directory()

                self.example_dir = os.path.join(self.data_dir, 'a_examples')
                if not os.path.isdir(self.example_dir):
                    os.mkdir(self.example_dir)

    def _root_dir_name(self) -> str:
        """To be overriden to specify root directory"""
        return os.getcwd()

    def _data_tree(self) -> str:
        """To be overriden to add more complexity to directory structure"""
        return ''

    def _exclude_save_keys(self) -> List:
        """Defines list of keys to be excluded when saving config. Meant to be overriden.

        Returns
        -------
        List: List of keys to be excluded when saving config
        """
        return []

    def _exclude_load_keys(self) -> List:
        """Defines list of keys to be excluded when loading config. Meant to be overriden.

        Returns
        -------
        List: List of keys to be excluded when loading config
        """
        return []

    def _exclude_equality_test_keys(self) -> List:
        """Defines list of keys to be excluded when checking for equality. Meant to be overriden.

        Returns
        -------
        List: List of keys to be excluded when checking for equality
        """
        return []

    def _check_passed_in_directory(self, directory: Union[str, None]) -> str:
        """Ensures that passed in directory is not None

        Parameters
        ----------
        directory: (str) - Directory to check. If None, attempts to replace it with data_dir

        Returns
        -------
        directory: (str) - Original directory or data_dir

        """
        if directory is None:
            directory = self.data_dir
        if directory is None:
            raise ValueError
        else:
            return directory

    def _config_is_equal(self,
                         config: Dict) -> bool:
        """Checks that config is equivalent to config of DataClass

        Parameters
        ----------
        config: (Dict) - Config to be checked for equality

        Returns
        -------
        bool: True if configs are equal, False otherwise

        """
        for key in self.config:
            if key not in self._exclude_equality_test_keys():
                if key not in config:
                    return False
                if config[key] != self.config[key]:
                    return False
        return True

    def _read_config_from_yaml_file(self,
                                    directory: str = None) -> Dict:
        """Reads config from a yaml file

        Parameters
        ----------
        directory: (str) - Directory in which config.yaml is stored. If None, looks in data_dir

        Returns
        -------
        settings: (Dict) - Returns a dict of the config contained in the yaml file

        """
        directory = self._check_passed_in_directory(directory)
        with open(os.path.join(directory, 'config.yaml'), 'r') as config_file:
            config = yaml.safe_load(config_file)
        for key in self._exclude_load_keys():
            config.pop(key)
        return config

    def _write_config_to_yaml_file(self,
                                   directory: str = None):
        """Writes config to yaml file in directory, except for number of images we have already simulated.

        Parameters
        ----------
        directory: (str) - Directory that simulation config will be written to. If None, writes to data_dir
        """
        directory = self._check_passed_in_directory(directory)
        with open(os.path.join(directory, 'config.yaml'), 'w') as config_file:
            tmp = {}
            for key in self._exclude_save_keys():
                tmp[key] = self.config.pop(key)
            yaml.dump(self.config, config_file)
            for key in tmp:
                self.config[key] = tmp[key]

    def _data_dir_surname(self) -> str:
        """Defines surname of data directory"""
        return 'data'

    def _find_or_make_data_directory(self,
                                     directory: str = None) -> str:
        """Attempts to find data directory in directory. Creates it if not found

        Parameters
        ----------
        directory: (str) - Directory that will be searched, defaults to base_dir

        Returns
        -------
        str: Location of data directory

        """
        if directory is None:
            directory = self.base_dir
        for data_dir in os.listdir(directory):
            if data_dir != 'config.yaml':
                config = self._read_config_from_yaml_file(os.path.join(directory, data_dir))
                if self._config_is_equal(config):
                    return os.path.join(directory, data_dir)
        data_dir = f'{directory}/{self._data_dir_surname()}_{len(os.listdir(directory)):04d}'
        os.mkdir(data_dir)
        self._write_config_to_yaml_file(data_dir)
        return data_dir

    def _retrieve_file_from_id_and_datatype(self,
                                            id_num: int,
                                            datatype: str) -> str:
        """Retrieves filename for id and datatype

        Parameters
        ----------
        id_num: (int) - ID number of file
        datatype: (str) - Datatype of file

        Returns
        -------
        str: Filename given ID and datatype

        """
        if datatype == 'popgen_image':
            return f'popgen_image_{id_num:09d}.npz'
        elif datatype == 'popgen_positions':
            return f'popgen_positions_{id_num:09d}.npy'
        else:
            raise NotImplementedError

    def _data_type_from_filename(self,
                                 filename: str) -> str:
        """Finds datatype from filename

        Parameters
        ----------
        filename: (str) - Name of file. Can be full path or last portion

        Returns
        -------
        str: datatype of file
        """
        file = os.path.basename(os.path.normpath(filename))
        if 'popgen_image' in file:
            if 'npz' in file:
                return 'popgen_image'
            elif 'npy' in file:
                return 'popgen_numpy_image'
            else:
                raise NotImplementedError
        elif 'popgen_positions' in file:
            return 'popgen_positions'
        else:
            raise NotImplementedError

    def _load_data(self,
                   file_path: str) -> np.ndarray:
        """Loads data from file_path

        Parameters
        ----------
        file_path: (str) - Path to file to be loaded

        Returns
        -------
        np.ndarray: Data in numpy array form

        """
        datatype = self._data_type_from_filename(file_path)
        if datatype == 'popgen_image':
            image = sparse.load_npz(file_path)
            return image.toarray()
        elif datatype == 'popgen_positions':
            return np.load(file_path)
        else:
            raise NotImplementedError

    def load_data(self,
                  full_file_path: str = None,
                  file: str = None,
                  id_num: int = None,
                  datatype: str = None,
                  directory: str = None) -> np.ndarray:
        """

        Parameters
        ----------
        full_file_path: (str) - Complete path to file
        file: (str) - If full_file_path is None, name of file to be loaded from directory or data_dir
            if directory is not specified
        id_num: (str) - If full_file_path and file are None, id of file to be loaded from directory or data_dir
            if directory is not specified
        datatype: (str) - If full_file_path and file are None, datatype of file to be loaded from directory or data_dir
            if directory is not specified
        directory: (str) - If full_file_path is None, directory to load file from. Loads from data_dir if None.

        Returns
        -------
        np.ndarray: Data loaded from specified file

        """
        if full_file_path is None:
            if file is None:
                if id is not None and datatype is not None:
                    file = self._retrieve_file_from_id_and_datatype(id_num, datatype)
                else:
                    raise Exception
            if directory is not None:
                full_file_path = os.path.join(directory, file)
            else:
                full_file_path = os.path.join(self.data_dir, file)
        return self._load_data(full_file_path)

    def save_data(self,
                  data: np.ndarray,
                  id_num: int,
                  datatype: str,
                  directory: str = None):
        """Saves data given numpy  matrix, id, and datatype

        Parameters
        ----------
        data: (np.ndarray) - Data to be saved
        id_num: (int) - ID number of data to be saved
        datatype: (str) - Datatype of data
        directory: (str) - Saves data to specified directory. If None, saves to data_dir
        """
        file = self._retrieve_file_from_id_and_datatype(id_num, datatype)
        directory = self._check_passed_in_directory(directory)
        full_file_path = os.path.join(directory, file)
        if datatype == 'popgen_image':
            sparse_matrix = sparse.csr_matrix(data)
            sparse.save_npz(full_file_path, sparse_matrix, compressed=True)
        elif datatype == 'popgen_positions':
            np.save(full_file_path, data)
        else:
            raise NotImplementedError

    def _last_saved_id(self,
                       datatype: str,
                       directory: str=None) -> int:
        """Looks in directory or data_dir and finds ID of last saved file of type datatype

        Parameters
        ----------
        datatype: (str) - datatype of files to search through
        directory: (str) - directory to search through, data_dir if None

        Returns
        -------
        int: ID of last saved file
        """
        directory = self._check_passed_in_directory(directory)
        id_num = 1
        while True:
            if not os.path.isfile(os.path.join(directory, self._retrieve_file_from_id_and_datatype(id_num, datatype))):
                return id_num-1
            id_num += 1

    def get_filenames(self,
                      datatype: str,
                      n: int,
                      directory: str=None) -> List[str]:
        """

        Parameters
        ----------
        datatype: (str) - Datatype of files to retrieve names of
        n: (int) - Number of files to retrieve names of
        directory: (str) - Directory to retrive filenames from, data_dir if None

        Returns
        -------
        List[str] - list of n filenames with type datatype from directory or data_dir

        """
        directory = self._check_passed_in_directory(directory)
        filenames = []
        for i in range(n):
            filenames.append(os.path.join(directory, self._retrieve_file_from_id_and_datatype(i+1, datatype)))
        return filenames

    def plot_example_image(self,
                           data: np.ndarray,
                           directory: str = None,
                           name: str = 'example'):
        """Plots example image in directory as png file

        Parameters
        ----------
        data: (np.ndarray) - numpy 2d binary matrix to be plotted
        directory: (str) - Directory to save example image, example_dir if None
        name: (str) - name of file to be saved
        """
        if directory is None:
            directory = self.example_dir
        im = Image.fromarray(np.uint8(255 * data))
        name = name + '.png'
        im.save(os.path.join(directory, name))





