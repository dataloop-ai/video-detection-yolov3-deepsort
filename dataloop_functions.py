import dtlpy as dl
import os


def push_package(project_name, package_name):
    project = dl.projects.get(project_name=project_name)
    inpt = dl.FunctionIO(type='Item', name='item')
    func = dl.PackageFunction(inputs=inpt)
    module = dl.PackageModule(functions=func)
    package = project.packages.push(package_name=package_name,
                                    modules=module,
                                    src_path=os.getcwd())
    print('Package pushed!')
    package.print()


def upload_artifacts(project_name, package_name):
    project = dl.projects.get(project_name=project_name)
    project.artifacts.upload(filepath='model_data/yolo.h5',
                             package_name=package_name)
    project.artifacts.upload(filepath='model_data/yolo_anchors.txt',
                             package_name=package_name)
    project.artifacts.upload(filepath='model_data/coco_classes.txt',
                             package_name=package_name)
    project.artifacts.upload(filepath='model_data\mars-small128.pb',
                             package_name=package_name)


def deploy_service(project_name, package_name):
    project = dl.projects.get(project_name=project_name)
    package = project.packages.get(package_name=package_name)
    service = package.services.deploy(service_name=package.name,
                                      sdk_version='1.15.6',
                                      init_input=[dl.FunctionIO(type=dl.PackageInputType.JSON,
                                                                name='package_name',
                                                                value=package_name)],
                                      runtime={'gpu': True,
                                               'numReplicas': 1,
                                               'concurrency': 1
                                               })
    print('Service deployed!')
    service.print()
    print(service.status())


def upload_service(package_name, service_name):
    package = dl.packages.get(package_name=package_name)
    service = package.services.get(service_name=service_name)
    service.package_revision = package.version
    service.update()


def create_trigger(service_name):
    service = dl.services.get(service_name=service_name)
    trigger = service.triggers.create(name=service.name,
                                      execution_mode=dl.TriggerExecutionMode.ONCE,
                                      resource='Item',
                                      actions=['Created'],
                                      filters={'$and': [{'dir': '/incoming'}]})
    print('Trigger created!')
    trigger.print()


def execute(service_name, item_id):
    service = dl.services.get(service_name=service_name)
    service.execute(function_name='run',
                    sync=True,
                    stream_logs=True,
                    execution_input=dl.FunctionIO(type=dl.PackageInputType.ITEM,
                                                  name='item',
                                                  value={'item_id': item_id}))

