# SOURCE_GROUP commands for the resource-folder structure (mainly MSVC) in applications and examples

# Required Vars (optional) for MSVC project groups:
# ${MY_ICE_FILES}
# ${MY_MODEL_FILES}
# ${MY_SHADER_FILES}
# ${MY_UI_FILES}
# ${MY_XML_FILES}

# define group names
SET(RESOURCES_GROUP "Resources")
SET(ICE_GROUP "Ice")
SET(MODEL_GROUP "Model")
SET(SHADERS_GROUP "Shader")
SET(UI_GROUP "Ui")
SET(XML_GROUP "Xml")

# and set everything to a subfolder of RESOURCES_GROUP


SOURCE_GROUP(
    ${RESOURCES_GROUP}\\${ICE_GROUP}
    FILES ${MY_ICE_FILES}
)

SOURCE_GROUP(
    ${RESOURCES_GROUP}\\${MODEL_GROUP}
    FILES ${MY_ICE_FILES}
)

SOURCE_GROUP(
    ${RESOURCES_GROUP}\\${SHADERS_GROUP}
    FILES ${MY_SHADER_FILES}
)

SOURCE_GROUP(
    ${RESOURCES_GROUP}\\${UI_GROUP}
    FILES ${MY_UI_FILES}
)

SOURCE_GROUP(
    ${RESOURCES_GROUP}\\${XML_GROUP}
    FILES ${MY_XML_FILES}
)



