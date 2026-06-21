from setuptools import find_packages,setup


hyphen_e_dot="-e ."

def get_requirements(file_path:str)->list[str]:
    '''
        this function return a list of requirements 
    '''

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)

    return requirements



setup(
    name="Customer chrun ml",
    version="0.0.1",
    author="Anandu Krishna K S",
    author_email="anandukrishna427@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)