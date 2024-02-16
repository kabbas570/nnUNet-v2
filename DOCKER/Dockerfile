#FROM pytorch/pytorch:1.11.1-cuda11.7-cudnn8-runtime

#FROM pytorch/pytorch:latest

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
FROM python:3.9-slim
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output /output/images/ /output/images/aorta-segmentation/ \
    && chown user:user /opt/app /input /output /output/images/ /output/images/aorta-segmentation/

USER user
WORKDIR /opt/app


ENV PATH="/home/user/.local/bin:${PATH}"



RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

RUN export MKL_SERVICE_FORCE_INTEL=1

RUN mkdir -p /opt/app/results/ \
    && chown user:user /opt/app/results/
COPY --chown=user:user results/ /opt/app/results/

RUN mkdir -p /input/images/ct/ \
    && chown user:user /input/images/ct/

COPY --chown=user:user test/ /input/images/ct/

COPY --chown=user:user requirements.txt /opt/app/
#RUN python -m piptools sync requirements.txt


#RUN python -m pip sync requirements.txt
RUN python -m pip install --user -r requirements.txt

COPY --chown=user:user process.py /opt/app/



COPY --chown=user:user nnUNet/ /opt/app/nnUNet/
RUN cd /opt/app/nnUNet/ && python -m pip install -e .
RUN cd ..

#ENTRYPOINT [ "python", "-m", "process" ]
ENTRYPOINT python -m process $0 $@
