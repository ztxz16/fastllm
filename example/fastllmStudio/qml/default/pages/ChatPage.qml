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
        if (chatViewModel) runningModels = chatViewModel.getRunningModels()
    }

    Component.onCompleted: refreshModels()
    onVisibleChanged: { if (visible) refreshModels() }

    Connections {
        target: repoViewModel
        function onModelsChanged() { chatPage.refreshModels() }
    }
    Connections {
        target: chatViewModel
        function onMessagesChanged() { chatListView.positionViewAtEnd() }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ── Top toolbar ──
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
                    text: qsTr("Local Chat")
                    font.pixelSize: Styles.Theme.fontSizeLarge
                    font.weight: Font.DemiBold
                    color: Styles.Theme.textBright
                }

                Item { Layout.preferredWidth: Styles.Theme.spacing }

                // Model selector
                Rectangle {
                    Layout.preferredWidth: 280; height: 34; radius: Styles.Theme.borderRadius
                    color: Styles.Theme.bgInput; border.color: Styles.Theme.border; border.width: 1

                    ComboBox {
                        id: modelSelector
                        anchors.fill: parent
                        palette.window: Styles.Theme.bgInput
                        palette.text: Styles.Theme.textPrimary
                        palette.buttonText: Styles.Theme.textPrimary

                        model: {
                            var items = [qsTr("Select a model")]
                            for (var i = 0; i < runningModels.length; i++)
                                items.push(runningModels[i].name + " (:" + runningModels[i].port + ")")
                            return items
                        }
                        currentIndex: 0
                        onActivated: function(idx) {
                            if (idx > 0 && idx - 1 < runningModels.length) {
                                selectedModelId = runningModels[idx - 1].model_id
                                if (chatViewModel) chatViewModel.selectModel(selectedModelId)
                            } else { selectedModelId = "" }
                        }
                    }
                }

                Item { Layout.fillWidth: true }

                // New chat
                Rectangle {
                    width: newChatRow.implicitWidth + 24; height: 32; radius: Styles.Theme.borderRadius
                    color: newChatMa.containsMouse ? Styles.Theme.bgInput : "transparent"
                    Row {
                        id: newChatRow; anchors.centerIn: parent; spacing: 6
                        Text { text: "+"; font.pixelSize: 15; font.weight: Font.Bold; color: Styles.Theme.textPrimary }
                        Text { text: qsTr("New Chat"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textPrimary }
                    }
                    MouseArea { id: newChatMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: { if (chatViewModel) chatViewModel.clearChat() } }
                }

                // Settings gear
                Rectangle {
                    width: 34; height: 34; radius: Styles.Theme.borderRadius
                    color: gearMa.containsMouse ? Styles.Theme.bgInput : "transparent"
                    Text { anchors.centerIn: parent; text: "⚙"; font.pixelSize: 18; color: Styles.Theme.textSecondary }
                    MouseArea { id: gearMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: samplingSettings.open() }
                }
            }

            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Styles.Theme.border }
        }

        // ── Empty state ──
        Item {
            Layout.fillWidth: true; Layout.fillHeight: true
            visible: selectedModelId === ""

            Column {
                anchors.centerIn: parent; spacing: 12

                Text { anchors.horizontalCenter: parent.horizontalCenter; text: "💬"; font.pixelSize: 48 }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: qsTr("Please start a model first")
                    font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textSecondary
                }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: qsTr("Select a model") + " ↑"
                    font.pixelSize: Styles.Theme.fontSizeSmall; color: Styles.Theme.textDisabled
                }
            }
        }

        // ── Chat area ──
        Rectangle {
            Layout.fillWidth: true; Layout.fillHeight: true
            color: Styles.Theme.chatBackground
            visible: selectedModelId !== ""

            ListView {
                id: chatListView
                anchors.fill: parent
                anchors.topMargin: Styles.Theme.paddingSmall
                anchors.bottomMargin: Styles.Theme.paddingSmall
                spacing: 2; clip: true
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

        // ── Input bar ──
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 56
            color: Styles.Theme.bgToolbar
            visible: selectedModelId !== ""

            Rectangle { anchors.top: parent.top; width: parent.width; height: 1; color: Styles.Theme.border }

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: Styles.Theme.paddingMedium
                anchors.rightMargin: Styles.Theme.paddingMedium
                anchors.topMargin: Styles.Theme.paddingSmall
                anchors.bottomMargin: Styles.Theme.paddingSmall
                spacing: Styles.Theme.spacingSmall

                // Input field
                Rectangle {
                    Layout.fillWidth: true; Layout.fillHeight: true
                    radius: Styles.Theme.borderRadius
                    color: Styles.Theme.bgInput
                    border.color: inputFld.activeFocus ? Styles.Theme.borderFocus : Styles.Theme.border
                    border.width: 1

                    TextInput {
                        id: inputFld
                        anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12
                        verticalAlignment: TextInput.AlignVCenter
                        color: Styles.Theme.textPrimary; selectionColor: Styles.Theme.accentColor
                        font.pixelSize: Styles.Theme.fontSizeMedium; clip: true
                        enabled: !chatViewModel || !chatViewModel.streaming
                        onAccepted: sendAction()

                        Text {
                            anchors.fill: parent; verticalAlignment: Text.AlignVCenter
                            text: qsTr("Type a message...")
                            color: Styles.Theme.textDisabled; font.pixelSize: Styles.Theme.fontSizeMedium
                            visible: !inputFld.text && !inputFld.activeFocus
                        }
                    }
                }

                // Send button
                Rectangle {
                    width: sendLbl.implicitWidth + 24; Layout.fillHeight: true
                    radius: Styles.Theme.borderRadius
                    color: {
                        var canSend = inputFld.text.trim() !== "" && (!chatViewModel || !chatViewModel.streaming)
                        return canSend ? (sendMa.containsMouse ? Styles.Theme.primaryHover : Styles.Theme.primaryColor)
                                       : Styles.Theme.bgInput
                    }

                    Text {
                        id: sendLbl; anchors.centerIn: parent
                        text: qsTr("Send"); font.pixelSize: Styles.Theme.fontSizeMedium
                        color: (inputFld.text.trim() !== "" && (!chatViewModel || !chatViewModel.streaming))
                               ? Styles.Theme.textOnAccent : Styles.Theme.textDisabled
                    }

                    MouseArea {
                        id: sendMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        onClicked: sendAction()
                    }
                }
            }
        }
    }

    function sendAction() {
        if (chatViewModel && inputFld.text.trim() !== "") {
            chatViewModel.sendMessage(inputFld.text)
            inputFld.text = ""
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
