import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles
import "../components"

Rectangle {
    id: chatPage
    color: Styles.Theme.bgEditor

    property var runningModels: []
    property string selectedModelId: ""

    function refreshModels() {
        if (!chatViewModel) return
        runningModels = chatViewModel.getRunningModels()

        var exists = false
        for (var i = 0; i < runningModels.length; i++) {
            if (runningModels[i].model_id === selectedModelId) {
                exists = true
                break
            }
        }
        if (!exists) {
            selectedModelId = ""
            modelSelector.currentIndex = 0
            chatViewModel.selectModel("")
        }
    }

    function modelOptions() {
        var items = [qsTr("Select deployed model")]
        for (var i = 0; i < runningModels.length; i++) {
            items.push(runningModels[i].name + "  127.0.0.1:" + runningModels[i].port)
        }
        return items
    }

    function sendAction() {
        var text = inputArea.text.trim()
        if (!chatViewModel || !text || !selectedModelId || chatViewModel.streaming) return
        chatViewModel.sendMessage(text)
        inputArea.text = ""
    }

    Component.onCompleted: refreshModels()
    onVisibleChanged: { if (visible) refreshModels() }

    Connections {
        target: repoViewModel
        function onModelsChanged() { chatPage.refreshModels() }
    }
    Connections {
        target: chatViewModel
        function onMessagesChanged() { Qt.callLater(function() { chatListView.positionViewAtEnd() }) }
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
                spacing: 12

                ColumnLayout {
                    spacing: 0
                    Layout.fillWidth: true

                    Text {
                        text: qsTr("Conversation")
                        color: Styles.Theme.textDisabled
                        font.pixelSize: Styles.Theme.fontSizeTiny
                        font.weight: Font.Bold
                    }
                    Text {
                        text: qsTr("Chat")
                        color: Styles.Theme.textBright
                        font.pixelSize: Styles.Theme.fontSizeLarge
                        font.weight: Font.DemiBold
                    }
                }

                ComboBox {
                    id: modelSelector
                    Layout.preferredWidth: 330
                    model: chatPage.modelOptions()
                    palette.window: Styles.Theme.bgInput
                    palette.text: Styles.Theme.textPrimary
                    palette.buttonText: Styles.Theme.textPrimary
                    onActivated: function(idx) {
                        if (idx > 0 && idx - 1 < runningModels.length) {
                            selectedModelId = runningModels[idx - 1].model_id
                            if (chatViewModel) chatViewModel.selectModel(selectedModelId)
                        } else {
                            selectedModelId = ""
                            if (chatViewModel) chatViewModel.selectModel("")
                        }
                    }
                }

                Rectangle {
                    Layout.preferredWidth: newChatLabel.implicitWidth + 24
                    Layout.preferredHeight: 32
                    radius: Styles.Theme.borderRadius
                    color: newChatMouse.containsMouse ? Styles.Theme.bgCardHover : "transparent"
                    border.color: Styles.Theme.border

                    Text {
                        id: newChatLabel
                        anchors.centerIn: parent
                        text: qsTr("New Chat")
                        color: Styles.Theme.textPrimary
                        font.pixelSize: Styles.Theme.fontSizeMedium
                    }

                    MouseArea {
                        id: newChatMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: { if (chatViewModel) chatViewModel.clearChat() }
                    }
                }

                Rectangle {
                    Layout.preferredWidth: 32
                    Layout.preferredHeight: 32
                    radius: Styles.Theme.borderRadius
                    color: settingsMouse.containsMouse ? Styles.Theme.bgCardHover : "transparent"
                    border.color: Styles.Theme.border

                    Text {
                        anchors.centerIn: parent
                        text: "S"
                        color: Styles.Theme.textSecondary
                        font.pixelSize: 12
                        font.weight: Font.Bold
                    }

                    MouseArea {
                        id: settingsMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: samplingSettings.open()
                    }
                }
            }

            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Styles.Theme.border }
        }

        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            Rectangle {
                anchors.fill: parent
                color: Styles.Theme.chatBackground
                visible: selectedModelId !== ""

                ListView {
                    id: chatListView
                    anchors.fill: parent
                    anchors.leftMargin: Math.min(80, width * 0.08)
                    anchors.rightMargin: Math.min(80, width * 0.08)
                    anchors.topMargin: 28
                    anchors.bottomMargin: 24
                    spacing: 4
                    clip: true
                    model: chatViewModel ? chatViewModel.messageList : null

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded
                        contentItem: Rectangle { implicitWidth: 6; radius: 3; color: Styles.Theme.scrollbar }
                    }

                    delegate: ChatBubble {
                        width: chatListView.width
                        role: model.role
                        content: model.content
                    }

                    onCountChanged: Qt.callLater(function() { chatListView.positionViewAtEnd() })
                }
            }

            Column {
                anchors.centerIn: parent
                spacing: 10
                visible: selectedModelId === ""

                Rectangle {
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: 42
                    height: 42
                    radius: 8
                    color: Styles.Theme.bgCard
                    border.color: Styles.Theme.border
                    Text {
                        anchors.centerIn: parent
                        text: "C"
                        color: Styles.Theme.textSecondary
                        font.pixelSize: 18
                        font.weight: Font.Bold
                    }
                }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: runningModels.length > 0 ? qsTr("Select deployed model") : qsTr("No deployed models")
                    font.pixelSize: Styles.Theme.fontSizeMedium
                    color: Styles.Theme.textSecondary
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 118
            color: Styles.Theme.bgToolbar
            visible: selectedModelId !== ""

            Rectangle { anchors.top: parent.top; width: parent.width; height: 1; color: Styles.Theme.border }

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20
                anchors.topMargin: 12
                anchors.bottomMargin: 12
                spacing: 10

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: Styles.Theme.borderRadiusLarge
                    color: Styles.Theme.bgInput
                    border.color: inputArea.activeFocus ? Styles.Theme.borderFocus : Styles.Theme.borderLight

                    TextArea {
                        id: inputArea
                        anchors.fill: parent
                        anchors.margins: 10
                        wrapMode: TextEdit.Wrap
                        color: Styles.Theme.textBright
                        placeholderText: qsTr("Send a message to the model...")
                        placeholderTextColor: Styles.Theme.textDisabled
                        selectionColor: Styles.Theme.accentColor
                        enabled: !chatViewModel || !chatViewModel.streaming
                        background: Rectangle { color: "transparent" }
                    }
                }

                Rectangle {
                    Layout.preferredWidth: 48
                    Layout.preferredHeight: 48
                    Layout.alignment: Qt.AlignBottom
                    radius: 24
                    color: {
                        var canSend = selectedModelId !== "" && inputArea.text.trim() !== "" && (!chatViewModel || !chatViewModel.streaming)
                        return canSend ? (sendMouse.containsMouse ? Styles.Theme.primaryHover : Styles.Theme.primaryColor)
                                       : Styles.Theme.bgCardHover
                    }

                    Text {
                        anchors.centerIn: parent
                        text: ">"
                        color: inputArea.text.trim() !== "" ? Styles.Theme.textOnAccent : Styles.Theme.textDisabled
                        font.pixelSize: 20
                        font.weight: Font.Bold
                    }

                    MouseArea {
                        id: sendMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: sendAction()
                    }
                }
            }
        }
    }

    SettingsPopup {
        id: samplingSettings
        anchors.centerIn: parent
        onSettingsApplied: function(temp, tp, tk, rp) {
            if (chatViewModel) {
                chatViewModel.setTemperature(temp)
                chatViewModel.setTopP(tp)
                chatViewModel.setTopK(tk)
                chatViewModel.setRepeatPenalty(rp)
            }
        }
    }
}
