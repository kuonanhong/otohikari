from git import Repo, InvalidGitRepositoryError

class DirtyGitRepositoryError(Exception):
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)

def get_git_hash(directory='.', length=7):
    '''
    This function will check the state of the git repository.

    * If there is no repo, an InvalidGitRepositoryError is raised.
    * If the repo is dirty, a DirtyRepositoryException is raised.
    * If none of the above applies, the hash of the HEAD commit is returned.

    Parameters
    ----------
    directory: str, optional
        The path to the directory to check.
    length: int, optional
        The number of characters of the hash to return.
    '''

    # Check the state of the github repository
    repo = Repo(directory, search_parent_directories=True)
    if repo.is_dirty():
        raise DirtyGitRepositoryError('The git repo has uncommited modifications. Aborting.')
    else:
        return repo.head.commit.hexsha[:length]

