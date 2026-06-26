import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../styles" as Styles
import "../components"

Rectangle {
    id: deployPage
    color: Styles.Theme.bgEditor

    function urlToPath(url) {
        var value = url.toString()
        if (value.indexOf("file://") === 0) {
            value = value.substring(7)
        }
        return decodeURIComponent(value)
    }

    function inferredName(path) {
        var cleaned = path.replace(/\\/g, "/").replace(/\/+$/, "")
        if (!cleaned) return ""
        var parts = cleaned.split("/")
        return parts[parts.length - 1]
    }

    function ensureNameFromPath() {
        if (!displayNameField.text.trim()) {
            displayNameField.text = inferredName(modelPathField.text.trim())
        }
    }

    function submitConfig(deploy) {
        var path = modelPathField.text.trim()
        if (!path) {
            formHint.text = qsTr("Model path is required.")
            return
        }

        ensureNameFromPath()
        var name = displayNameField.text.trim()
        var apiName = apiModelNameField.text.trim()
        var device = deviceCombo.currentText
        var threads = threadsSpin.value
        var dtype = dtypeCombo.currentText
        var port = portSpin.value

        if (deploy) {
            if (repoViewModel) repoViewModel.deployModelConfig(name, path, device, apiName, threads, dtype, port)
            formHint.text = qsTr("Deploy command submitted.")
        } else {
            if (repoViewModel) repoViewModel.addModelConfig(name, path, device, apiName, threads, dtype, port)
            formHint.text = qsTr("Configuration saved.")
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: Styles.Theme.toolbarHeight
            color: Styles.Theme.bgToolbar

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 18
                anchors.rightMargin: 18
                spacing: Styles.Theme.spacing

                ColumnLayout {
                    spacing: 0
                    Layout.fillWidth: true

                    Text {
                        text: qsTr("OpenAI Compatible")
                        color: Styles.Theme.textDisabled
                        font.pixelSize: Styles.Theme.fontSizeTiny
                        font.weight: Font.Bold
                    }
                    Text {
                        text: qsTr("Model Deploy")
                        color: Styles.Theme.textBright
                        font.pixelSize: Styles.Theme.fontSizeLarge
                        font.weight: Font.DemiBold
                    }
                }

                Rectangle {
                    Layout.preferredWidth: runningLabel.implicitWidth + 18
                    Layout.preferredHeight: 26
                    radius: 13
                    color: Styles.Theme.stoppedBadgeBg

                    Text {
                        id: runningLabel
                        anchors.centerIn: parent
                        text: qsTr("Local API")
                        color: Styles.Theme.textSecondary
                        font.pixelSize: Styles.Theme.fontSizeTiny
                        font.weight: Font.DemiBold
                    }
                }
            }

            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Styles.Theme.border }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 18
            spacing: 18

            Rectangle {
                Layout.preferredWidth: 378
                Layout.fillHeight: true
                radius: Styles.Theme.borderRadiusLarge
                color: Styles.Theme.bgCard
                border.color: Styles.Theme.border

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 12

                    Text {
                        text: qsTr("Model Configuration")
                        color: Styles.Theme.textBright
                        font.pixelSize: 17
                        font.weight: Font.DemiBold
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Text { text: qsTr("Model path"); color: Styles.Theme.textSecondary; font.pixelSize: Styles.Theme.fontSizeSmall }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            TextField {
                                id: modelPathField
                                Layout.fillWidth: true
                                color: Styles.Theme.textPrimary
                                placeholderText: "/models/qwen/model.flm"
                                placeholderTextColor: Styles.Theme.textDisabled
                                selectionColor: Styles.Theme.accentColor
                                onEditingFinished: ensureNameFromPath()
                                background: Rectangle {
                                    color: Styles.Theme.bgInput
                                    border.color: modelPathField.activeFocus ? Styles.Theme.borderFocus : Styles.Theme.border
                                    radius: Styles.Theme.borderRadius
                                }
                            }

                            Rectangle {
                                Layout.preferredWidth: 34
                                Layout.preferredHeight: 34
                                radius: Styles.Theme.borderRadius
                                color: fileMouse.containsMouse ? Styles.Theme.bgCardHover : Styles.Theme.bgInput
                                border.color: Styles.Theme.border
                                Text { anchors.centerIn: parent; text: "F"; color: Styles.Theme.textSecondary; font.pixelSize: 12; font.weight: Font.Bold }
                                MouseArea { id: fileMouse; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: modelFileDialog.open() }
                            }

                            Rectangle {
                                Layout.preferredWidth: 34
                                Layout.preferredHeight: 34
                                radius: Styles.Theme.borderRadius
                                color: folderMouse.containsMouse ? Styles.Theme.bgCardHover : Styles.Theme.bgInput
                                border.color: Styles.Theme.border
                                Text { anchors.centerIn: parent; text: "D"; color: Styles.Theme.textSecondary; font.pixelSize: 12; font.weight: Font.Bold }
                                MouseArea { id: folderMouse; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: modelFolderDialog.open() }
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Text { text: qsTr("Name"); color: Styles.Theme.textSecondary; font.pixelSize: Styles.Theme.fontSizeSmall }
                        TextField {
                            id: displayNameField
                            Layout.fillWidth: true
                            color: Styles.Theme.textPrimary
                            placeholderText: qsTr("Visible name")
                            placeholderTextColor: Styles.Theme.textDisabled
                            selectionColor: Styles.Theme.accentColor
                            background: Rectangle {
                                color: Styles.Theme.bgInput
                                border.color: displayNameField.activeFocus ? Styles.Theme.borderFocus : Styles.Theme.border
                                radius: Styles.Theme.borderRadius
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        Text { text: "model_name"; color: Styles.Theme.textSecondary; font.pixelSize: Styles.Theme.fontSizeSmall }
                        TextField {
                            id: apiModelNameField
                            Layout.fillWidth: true
                            color: Styles.Theme.textPrimary
                            placeholderText: qsTr("API model id")
                            placeholderTextColor: Styles.Theme.textDisabled
                            selectionColor: Styles.Theme.accentColor
                            background: Rectangle {
                                color: Styles.Theme.bgInput
                                border.color: apiModelNameField.activeFocus ? Styles.Theme.borderFocus : Styles.Theme.border
                                radius: Styles.Theme.borderRadius
                            }
                        }
                    }

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        columnSpacing: 10
                        rowSpacing: 10

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6
                            Text { text: qsTr("Device"); color: Styles.Theme.textSecondary; font.pixelSize: Styles.Theme.fontSizeSmall }
                            ComboBox {
                                id: deviceCombo
                                Layout.fillWidth: true
                                model: ["", "cpu", "cuda", "numa", "multicuda", "tops", "tfacc"]
                                palette.window: Styles.Theme.bgInput
                                palette.text: Styles.Theme.textPrimary
                                palette.buttonText: Styles.Theme.textPrimary
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6
                            Text { text: qsTr("Port"); color: Styles.Theme.textSecondary; font.pixelSize: Styles.Theme.fontSizeSmall }
                            SpinBox {
                                id: portSpin
                                Layout.fillWidth: true
                                from: 0
                                to: 65535
                                value: 8080
                                palette.window: Styles.Theme.bgInput
                                palette.text: Styles.Theme.textPrimary
                                palette.buttonText: Styles.Theme.textPrimary
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6
                            Text { text: qsTr("Threads"); color: Styles.Theme.textSecondary; font.pixelSize: Styles.Theme.fontSizeSmall }
                            SpinBox {
                                id: threadsSpin
                                Layout.fillWidth: true
                                from: 1
                                to: 256
                                value: 4
                                palette.window: Styles.Theme.bgInput
                                palette.text: Styles.Theme.textPrimary
                                palette.buttonText: Styles.Theme.textPrimary
                            }
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6
                            Text { text: qsTr("Dtype"); color: Styles.Theme.textSecondary; font.pixelSize: Styles.Theme.fontSizeSmall }
                            ComboBox {
                                id: dtypeCombo
                                Layout.fillWidth: true
                                model: ["auto", "float16", "float32", "int8", "int4"]
                                palette.window: Styles.Theme.bgInput
                                palette.text: Styles.Theme.textPrimary
                                palette.buttonText: Styles.Theme.textPrimary
                            }
                        }
                    }

                    Item { Layout.fillHeight: true }

                    Text {
                        id: formHint
                        Layout.fillWidth: true
                        text: qsTr("Ready")
                        color: Styles.Theme.textDisabled
                        font.pixelSize: Styles.Theme.fontSizeTiny
                        elide: Text.ElideRight
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Rectangle {
                            Layout.preferredWidth: saveLabel.implicitWidth + 24
                            Layout.preferredHeight: 32
                            radius: Styles.Theme.borderRadius
                            color: saveMouse.containsMouse ? Styles.Theme.bgCardHover : "transparent"
                            border.color: Styles.Theme.border

                            Text {
                                id: saveLabel
                                anchors.centerIn: parent
                                text: qsTr("Save Config")
                                color: Styles.Theme.textPrimary
                                font.pixelSize: Styles.Theme.fontSizeMedium
                            }

                            MouseArea {
                                id: saveMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: submitConfig(false)
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 32
                            radius: Styles.Theme.borderRadius
                            color: deployMouse.containsMouse ? Styles.Theme.primaryHover : Styles.Theme.primaryColor

                            Text {
                                anchors.centerIn: parent
                                text: qsTr("Deploy")
                                color: Styles.Theme.textOnAccent
                                font.pixelSize: Styles.Theme.fontSizeMedium
                                font.weight: Font.DemiBold
                            }

                            MouseArea {
                                id: deployMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: submitConfig(true)
                            }
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                radius: Styles.Theme.borderRadiusLarge
                color: Styles.Theme.bgCard
                border.color: Styles.Theme.border

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 12

                    RowLayout {
                        Layout.fillWidth: true
                        Text {
                            text: qsTr("Saved Configs")
                            color: Styles.Theme.textBright
                            font.pixelSize: 17
                            font.weight: Font.DemiBold
                            Layout.fillWidth: true
                        }
                        Rectangle {
                            Layout.preferredWidth: modelCountLabel.implicitWidth + 14
                            Layout.preferredHeight: 22
                            radius: 11
                            color: Styles.Theme.stoppedBadgeBg
                            Text {
                                id: modelCountLabel
                                anchors.centerIn: parent
                                text: modelListView.count
                                color: Styles.Theme.textSecondary
                                font.pixelSize: Styles.Theme.fontSizeTiny
                                font.weight: Font.Bold
                            }
                        }
                    }

                    ListView {
                        id: modelListView
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 8
                        clip: true
                        model: repoViewModel ? repoViewModel.modelList : null

                        ScrollBar.vertical: ScrollBar {
                            policy: ScrollBar.AsNeeded
                            contentItem: Rectangle { implicitWidth: 6; radius: 3; color: Styles.Theme.scrollbar }
                        }

                        delegate: ModelCard {
                            width: modelListView.width
                            modelId: model.modelId
                            modelName: model.name
                            modelPath: model.path
                            running: model.running
                            modelPort: model.port
                            device: model.device
                            apiModelName: model.apiModelName
                            configPort: model.configPort

                            onSettingsClicked: function(mid) {
                                settingsDialog.modelId = mid
                                settingsDialog.device = model.device
                                settingsDialog.threads = model.threads
                                settingsDialog.dtype = model.dtype
                                settingsDialog.configPort = model.configPort
                                settingsDialog.apiModelName = model.apiModelName
                                settingsDialog.open()
                            }
                            onStartClicked: function(mid) { if (repoViewModel) repoViewModel.startModel(mid) }
                            onStopClicked: function(mid) { if (repoViewModel) repoViewModel.stopModel(mid) }
                            onRemoveClicked: function(mid) { if (repoViewModel) repoViewModel.removeModel(mid) }
                        }

                        Column {
                            anchors.centerIn: parent
                            spacing: 8
                            visible: modelListView.count === 0

                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: "F"
                                font.pixelSize: 40
                                font.weight: Font.Bold
                                color: Styles.Theme.textDisabled
                            }
                            Text {
                                anchors.horizontalCenter: parent.horizontalCenter
                                text: qsTr("No configs")
                                font.pixelSize: Styles.Theme.fontSizeMedium
                                color: Styles.Theme.textSecondary
                            }
                        }
                    }
                }
            }
        }
    }

    FileDialog {
        id: modelFileDialog
        title: qsTr("Select model file")
        fileMode: FileDialog.OpenFile
        onAccepted: {
            modelPathField.text = deployPage.urlToPath(selectedFile)
            deployPage.ensureNameFromPath()
        }
    }

    FolderDialog {
        id: modelFolderDialog
        title: qsTr("Select model directory")
        onAccepted: {
            modelPathField.text = deployPage.urlToPath(selectedFolder)
            deployPage.ensureNameFromPath()
        }
    }

    ModelSettingsDialog {
        id: settingsDialog
        anchors.centerIn: parent
        onSaved: function(mid, device, threads, dtype, port, apiModelName) {
            if (repoViewModel) repoViewModel.updateLaunchConfig(mid, device, threads, dtype, port, apiModelName)
        }
    }
}
