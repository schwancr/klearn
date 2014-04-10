u"""
setup.py: Install klearn package for using kernel methods to learn!
"""
import os, sys
from glob import glob
import subprocess
import shutil
import tempfile
from distutils.ccompiler import new_compiler

VERSION = "0.1"
ISRELEASED = False
__author__ = "Christian Schwantes"
__version__ = VERSION

# metadata for setup()
metadata = {
    'version': VERSION,
    'author': __author__,
    'author_email': 'schwancr@stanford.edu',
    'platforms': ["Linux", "Mac OS X"],
    'zip_safe': False,
    'description': "Python Code for Learning with Kernels",
    'long_description': """kernel learning for various datasets,
    currently only ktICA and kPCA are implemented. To come: kCCA
    kFDA, etc. and maybe the non-kernelized versions..."""}


def hasfunction(cc, funcname, include=None, extra_postargs=None):
    # From http://stackoverflow.com/questions/
    #            7018879/disabling-output-when-compiling-with-distutils
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            if include is not None:
                f.write('#include %s\n' % include)
            f.write('int main(void) {\n')
            f.write('    %s;\n' % funcname)
            f.write('}\n')
            f.close()
            devnull = open(os.devnull, 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            objects = cc.compile([fname], output_dir=tmpdir,
                                 extra_postargs=extra_postargs)
            cc.link_executable(objects, os.path.join(tmpdir, 'a.out'))
        except Exception as e:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)

# Return the git revision as a string
# copied from numpy setup.py
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION

def write_version_py(filename='klearn/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM KLEARN SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version' : FULLVERSION,
                       'git_revision' : GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


if os.environ.get('READTHEDOCS', None) == 'True' and __name__ == '__main__':
    # On READTHEDOCS, the service that hosts our documentation, the build
    # environment does not have numpy and cannot build C extension modules,
    # so if we detect this environment variable, we're going to bail out
    # and run a minimal setup. This only installs the python packages, which
    # is not enough to RUN anything, but should be enough to introspect the
    # docstrings, which is what's needed for the documentation
    from distutils.core import setup
    import tempfile, shutil
    write_version_py()
    
    metadata['name'] = 'ktica'
    metadata['packages'] = ['ktica', 'ktica.objects', 'ktica.kernels', 'ktica.metrics']
    metadata['scripts'] = [e for e in glob('scripts/*.py') if not e.endswith('__.py')]

    # dirty, dirty trick to install "mock" packages
    mockdir = tempfile.mkdtemp()
    open(os.path.join(mockdir, '__init__.py'), 'w').close()
    metadata['package_dir'] = {'ktica': 'ktica',
                               'ktica.learners' : 'ktica/learners',
                               'ktica.kernels' : 'ktica/kernels',
                               'ktica.metrics' : 'ktica/metrics'}

    setup(**metadata)
    shutil.rmtree(mockdir) #clean up dirty trick
    sys.exit(1)


# now procede to standard setup
# setuptools needs to come before numpy.distutils to get install_requires
import setuptools 
import numpy
from distutils import sysconfig
from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='',top_path=None):
    "Configure the build"

    config = Configuration('klearn',
                           package_parent=parent_package,
                           top_path=top_path,
                           package_path='klearn')
    config.set_options(assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=False)
    
    #once all of the data is in one place, we can add it with this
    #config.add_data_dir('reference')
    
    # add the scipts, so they can be called from the command line
    config.add_scripts([e for e in glob('scripts/*.py') if not e.endswith('__.py')])
    
    # add scripts as a subpackage (so they can be imported from other scripts)
    #config.add_subpackage('scripts',
    #                      subpackage_path=None)

    config.add_subpackage('kernels',
                          subpackage_path='klearn/kernels')
    config.add_subpackage('msmbmetrics',
                          subpackage_path='klearn/msmbmetrics')
    config.add_subpackage('learners',
                          subpackage_path='klearn/learners')

    return config

if __name__ == '__main__':
    write_version_py()
    metadata['configuration'] = configuration
    setup(**metadata)
