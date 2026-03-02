import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "../styles" as Styles
import "../components"

Item {
    id: repoPage
    property bool showMarket: false

    StackLayout {
        anchors.fill: parent
        currentIndex: repoPage.showMarket ? 1 : 0

        // ── Page 0: Repository ──
        Rectangle {
            color: Styles.Theme.bgEditor

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                // Toolbar
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: Styles.Theme.toolbarHeight
                    color: Styles.Theme.bgToolbar

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: Styles.Theme.paddingLarge
                        anchors.rightMargin: Styles.Theme.paddingLarge
                        spacing: Styles.Theme.spacing

                        Text {
                            text: qsTr("Model Repository")
                            font.pixelSize: Styles.Theme.fontSizeLarge
                            font.weight: Font.DemiBold
                            color: Styles.Theme.textBright
                        }

                        Item { Layout.fillWidth: true }

                        // Add model button
                        Rectangle {
                            id: addBtn
                            width: addRow.implicitWidth + 28
                            height: 34; radius: Styles.Theme.borderRadius
                            color: addMa.containsMouse ? Styles.Theme.primaryHover : Styles.Theme.primaryColor

                            Row {
                                id: addRow; anchors.centerIn: parent; spacing: 6
                                Text { text: "+"; font.pixelSize: 15; font.weight: Font.Bold; color: Styles.Theme.textOnAccent }
                                Text { text: qsTr("Add Model"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textOnAccent }
                            }

                            MouseArea {
                                id: addMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                                onClicked: addPopup.open()
                            }
                        }
                    }

                    Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Styles.Theme.border }
                }

                // Model list
                ListView {
                    id: modelListView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.margins: Styles.Theme.paddingMedium
                    spacing: Styles.Theme.spacingSmall
                    clip: true
                    model: repoViewModel ? repoViewModel.modelList : null

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded
                        contentItem: Rectangle { implicitWidth: 6; radius: 3; color: Styles.Theme.scrollbar }
                    }

                    delegate: ModelCard {
                        width: modelListView.width - 2 * Styles.Theme.paddingMedium
                        anchors.horizontalCenter: parent ? parent.horizontalCenter : undefined
                        modelId: model.modelId
                        modelName: model.name
                        modelPath: model.path
                        running: model.running
                        modelPort: model.port

                        onSettingsClicked: function(mid) {
                            settingsDialog.modelId = mid
                            settingsDialog.device = model.device
                            settingsDialog.threads = model.threads
                            settingsDialog.dtype = model.dtype
                            settingsDialog.configPort = model.configPort
                            settingsDialog.open()
                        }
                        onStartClicked: function(mid) { if (repoViewModel) repoViewModel.startModel(mid) }
                        onStopClicked: function(mid) { if (repoViewModel) repoViewModel.stopModel(mid) }
                        onRemoveClicked: function(mid) { if (repoViewModel) repoViewModel.removeModel(mid) }
                    }

                    // Empty state
                    Column {
                        anchors.centerIn: parent
                        spacing: 8
                        visible: modelListView.count === 0

                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: "📦"
                            font.pixelSize: 40
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: qsTr("No models added yet")
                            font.pixelSize: Styles.Theme.fontSizeMedium
                            color: Styles.Theme.textSecondary
                        }
                        Text {
                            anchors.horizontalCenter: parent.horizontalCenter
                            text: qsTr("Add Model")
                            font.pixelSize: Styles.Theme.fontSizeSmall
                            color: Styles.Theme.textLink
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: addPopup.open()
                            }
                        }
                    }
                }
            }
        }

        // ── Page 1: Market ──
        ModelMarketPage {
            onBackClicked: repoPage.showMarket = false
            onModelDownloaded: {
                repoPage.showMarket = false
                if (repoViewModel) { repoViewModel.modelList.reload(); repoViewModel.modelsChanged() }
            }
        }
    }

    FolderDialog {
        id: folderDialog
        title: qsTr("Add Local Model")
        onAccepted: {
            var path = selectedFolder.toString()
            if (path.startsWith("file://")) path = path.substring(7)
            var name = path.split("/").pop()
            if (repoViewModel) repoViewModel.addLocalModel(name, path)
        }
    }

    ModelSettingsDialog {
        id: settingsDialog
        anchors.centerIn: parent
        onSaved: function(mid, device, threads, dtype, port) {
            if (repoViewModel) repoViewModel.updateLaunchConfig(mid, device, threads, dtype, port)
        }
    }

    Dialog {
        id: addPopup
        modal: true
        title: ""
        standardButtons: Dialog.NoButton
        anchors.centerIn: parent
        width: 360
        dim: true

        Overlay.modal: Rectangle { color: "#80000000" }

        background: Rectangle {
            color: Styles.Theme.bgPopup
            border.color: Styles.Theme.border
            radius: Styles.Theme.borderRadiusLarge
        }

        contentItem: ColumnLayout {
            spacing: 0

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 44
                color: "transparent"

                Text {
                    anchors.centerIn: parent
                    text: qsTr("Add Model")
                    font.pixelSize: Styles.Theme.fontSizeLarge
                    font.weight: Font.DemiBold
                    color: Styles.Theme.textBright
                }
                Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Styles.Theme.border }
            }

            Item { Layout.preferredHeight: Styles.Theme.paddingMedium }

            Repeater {
                model: [
                    { icon: "📁", label: QT_TR_NOOP("Add Local Model"), action: "local" },
                    { icon: "🌐", label: QT_TR_NOOP("Download from Market"), action: "market" }
                ]

                delegate: Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 52
                    Layout.leftMargin: Styles.Theme.paddingMedium
                    Layout.rightMargin: Styles.Theme.paddingMedium
                    radius: Styles.Theme.borderRadius
                    color: itemMa.containsMouse ? Styles.Theme.bgCardHover : Styles.Theme.bgCard
                    border.color: itemMa.containsMouse ? Styles.Theme.borderFocus : Styles.Theme.border
                    border.width: 1

                    Behavior on color { ColorAnimation { duration: Styles.Theme.animDuration } }
                    Behavior on border.color { ColorAnimation { duration: Styles.Theme.animDuration } }

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: Styles.Theme.paddingMedium
                        anchors.rightMargin: Styles.Theme.paddingMedium
                        spacing: Styles.Theme.spacing

                        Text {
                            text: modelData.icon
                            font.pixelSize: 22
                        }
                        Text {
                            text: qsTr(modelData.label)
                            font.pixelSize: Styles.Theme.fontSizeMedium
                            color: Styles.Theme.textBright
                            Layout.fillWidth: true
                        }
                        Text {
                            text: "→"
                            font.pixelSize: 14
                            color: Styles.Theme.textSecondary
                        }
                    }

                    MouseArea {
                        id: itemMa
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: {
                            addPopup.close()
                            if (modelData.action === "local") folderDialog.open()
                            else repoPage.showMarket = true
                        }
                    }
                }
            }

            Item { Layout.preferredHeight: Styles.Theme.paddingMedium }

            Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: Styles.Theme.border }

            RowLayout {
                Layout.fillWidth: true
                Layout.margins: Styles.Theme.paddingMedium

                Item { Layout.fillWidth: true }

                Rectangle {
                    width: cancelLbl.implicitWidth + 24; height: 30; radius: Styles.Theme.borderRadius
                    color: cancelMa.containsMouse ? Styles.Theme.bgInput : "transparent"
                    border.color: Styles.Theme.border; border.width: 1
                    Text { id: cancelLbl; anchors.centerIn: parent; text: qsTr("Cancel"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textPrimary }
                    MouseArea { id: cancelMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: addPopup.close() }
                }
            }
        }
    }
}
