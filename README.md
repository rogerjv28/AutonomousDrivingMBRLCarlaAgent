# AutonomousDrivingE2ECarlaAgent

Código del TFM: "Comparativa de Políticas de Conducción Autónoma: Enfoque vision-only contra fusión multisensor mediante Aprendizaje por Refuerzo"

## Uso

```bash
conda env create -f environment.yml
conda activate adcarla
pip install -e . # Instala el paquete adcarla


#   Windows (PowerShell):
.\CarlaUE4.exe -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
#   Linux:
./CarlaUE4.sh  -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
```
