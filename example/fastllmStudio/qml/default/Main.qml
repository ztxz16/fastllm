import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "styles" as Styles
import "pages"

ApplicationWindow {
    id: root
    visible: true
    width: 1240
    height: 780
    minimumWidth: 980
    minimumHeight: 620
    title: "Fastllm Studio"
    color: Styles.Theme.bgBase

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: Styles.Theme.sidebarWidth
            color: Styles.Theme.sidebarBg

            ColumnLayout {
                anchors.fill: parent
                anchors.topMargin: 10
                anchors.bottomMargin: 10
                spacing: 8

                Rectangle {
                    Layout.alignment: Qt.AlignHCenter
                    Layout.preferredWidth: 30
                    Layout.preferredHeight: 30
                    radius: 7
                    color: Styles.Theme.bgCard
                    border.color: Styles.Theme.borderLight

                    Rectangle {
                        anchors.centerIn: parent
                        width: 20
                        height: 20
                        radius: 5
                        color: Styles.Theme.accentColor

                        Text {
                            anchors.centerIn: parent
                            text: "F"
                            color: Styles.Theme.textOnAccent
                            font.pixelSize: 13
                            font.weight: Font.Bold
                        }
                    }
                }

                Repeater {
                    model: [
                        { icon: "D", label: QT_TR_NOOP("Deploy"), page: 0 },
                        { icon: "C", label: QT_TR_NOOP("Chat"), page: 1 }
                    ]

                    delegate: Rectangle {
                        Layout.alignment: Qt.AlignHCenter
                        Layout.preferredWidth: 38
                        Layout.preferredHeight: 38
                        radius: 7
                        color: pageStack.currentIndex === modelData.page || railMouse.containsMouse
                               ? Styles.Theme.sidebarItemActive : "transparent"

                        Rectangle {
                            anchors.left: parent.left
                            anchors.leftMargin: -6
                            anchors.verticalCenter: parent.verticalCenter
                            width: 3
                            height: 22
                            radius: 2
                            color: pageStack.currentIndex === modelData.page
                                   ? Styles.Theme.sidebarIndicator : "transparent"
                        }

                        Text {
                            anchors.centerIn: parent
                            text: modelData.icon
                            color: pageStack.currentIndex === modelData.page
                                   ? Styles.Theme.textBright : Styles.Theme.textSecondary
                            font.pixelSize: 13
                            font.weight: Font.DemiBold
                        }

                        MouseArea {
                            id: railMouse
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: pageStack.currentIndex = modelData.page

                            ToolTip.visible: railMouse.containsMouse
                            ToolTip.text: qsTr(modelData.label)
                            ToolTip.delay: 500
                        }
                    }
                }

                Item { Layout.fillHeight: true }

                Rectangle {
                    Layout.alignment: Qt.AlignHCenter
                    Layout.preferredWidth: 38
                    Layout.preferredHeight: 38
                    radius: 7
                    color: langMouse.containsMouse ? Styles.Theme.sidebarItemHover : "transparent"

                    Text {
                        anchors.centerIn: parent
                        text: "A"
                        color: Styles.Theme.textSecondary
                        font.pixelSize: 13
                        font.weight: Font.DemiBold
                    }

                    MouseArea {
                        id: langMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: langDialog.open()
                    }
                }
            }
        }

        Rectangle { Layout.fillHeight: true; Layout.preferredWidth: 1; color: Styles.Theme.border }

        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: Styles.Theme.sidebarExpandedWidth
            color: Styles.Theme.bgSidebar

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 82
                    Layout.leftMargin: 14
                    Layout.rightMargin: 14
                    Layout.topMargin: 16
                    spacing: 2

                    Text {
                        text: qsTr("Workspace")
                        color: Styles.Theme.textDisabled
                        font.pixelSize: Styles.Theme.fontSizeTiny
                        font.weight: Font.Bold
                    }
                    Text {
                        text: "Fastllm Studio"
                        color: Styles.Theme.textBright
                        font.pixelSize: 17
                        font.weight: Font.DemiBold
                    }
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.leftMargin: 8
                    Layout.rightMargin: 8
                    spacing: 6

                    Repeater {
                        model: [
                            { key: "01", label: QT_TR_NOOP("Model Deploy"), detail: QT_TR_NOOP("Configs and runtime"), page: 0 },
                            { key: "02", label: QT_TR_NOOP("Chat"), detail: QT_TR_NOOP("Running models"), page: 1 }
                        ]

                        delegate: Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 44
                            radius: Styles.Theme.borderRadius
                            color: pageStack.currentIndex === modelData.page || navMouse.containsMouse
                                   ? Styles.Theme.sidebarItemHover : "transparent"

                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: 10
                                anchors.rightMargin: 10
                                spacing: 10

                                Rectangle {
                                    Layout.preferredWidth: 7
                                    Layout.preferredHeight: 7
                                    radius: 4
                                    color: pageStack.currentIndex === modelData.page
                                           ? Styles.Theme.accentColor : Styles.Theme.textDisabled
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 0
                                    Text {
                                        text: qsTr(modelData.label)
                                        color: pageStack.currentIndex === modelData.page
                                               ? Styles.Theme.textBright : Styles.Theme.textPrimary
                                        font.pixelSize: Styles.Theme.fontSizeMedium
                                        font.weight: Font.DemiBold
                                        elide: Text.ElideRight
                                        Layout.fillWidth: true
                                    }
                                    Text {
                                        text: qsTr(modelData.detail)
                                        color: Styles.Theme.textDisabled
                                        font.pixelSize: Styles.Theme.fontSizeTiny
                                        elide: Text.ElideRight
                                        Layout.fillWidth: true
                                    }
                                }

                                Text {
                                    text: modelData.key
                                    color: Styles.Theme.textDisabled
                                    font.pixelSize: Styles.Theme.fontSizeTiny
                                }
                            }

                            MouseArea {
                                id: navMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                onClicked: pageStack.currentIndex = modelData.page
                            }
                        }
                    }
                }

                Item { Layout.fillHeight: true }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 34
                    color: "transparent"
                    border.color: Styles.Theme.border
                    border.width: 1

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 12
                        anchors.rightMargin: 12
                        spacing: 8

                        Rectangle {
                            Layout.preferredWidth: 7
                            Layout.preferredHeight: 7
                            radius: 4
                            color: Styles.Theme.successColor
                        }
                        Text {
                            text: qsTr("Local runtime")
                            color: Styles.Theme.textSecondary
                            font.pixelSize: Styles.Theme.fontSizeTiny
                            Layout.fillWidth: true
                        }
                        Text {
                            text: "ftllm"
                            color: Styles.Theme.infoColor
                            font.pixelSize: Styles.Theme.fontSizeTiny
                            font.weight: Font.DemiBold
                        }
                    }
                }
            }
        }

        Rectangle { Layout.fillHeight: true; Layout.preferredWidth: 1; color: Styles.Theme.border }

        StackLayout {
            id: pageStack
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: 0

            ModelRepoPage { id: modelRepoPage }
            ChatPage { id: chatPage }
        }
    }

    Dialog {
        id: langDialog
        modal: true
        title: ""
        standardButtons: Dialog.NoButton
        anchors.centerIn: parent
        width: 300
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
                    text: qsTr("Language")
                    font.pixelSize: Styles.Theme.fontSizeLarge
                    font.weight: Font.DemiBold
                    color: Styles.Theme.textBright
                }
                Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Styles.Theme.border }
            }

            Item { Layout.preferredHeight: Styles.Theme.paddingSmall }

            Repeater {
                model: [
                    { code: "zh_CN", label: "Simplified Chinese" },
                    { code: "en_US", label: "English" }
                ]

                delegate: Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 44
                    Layout.leftMargin: Styles.Theme.paddingMedium
                    Layout.rightMargin: Styles.Theme.paddingMedium
                    radius: Styles.Theme.borderRadius
                    color: langItemMouse.containsMouse ? Styles.Theme.bgCardHover : "transparent"

                    property bool isCurrent: langManager && langManager.currentLanguage === modelData.code

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: Styles.Theme.paddingMedium
                        anchors.rightMargin: Styles.Theme.paddingMedium
                        spacing: Styles.Theme.spacing

                        Text {
                            text: modelData.label
                            font.pixelSize: Styles.Theme.fontSizeMedium
                            color: isCurrent ? Styles.Theme.accentColor : Styles.Theme.textBright
                            Layout.fillWidth: true
                        }
                        Text {
                            text: isCurrent ? "OK" : ""
                            font.pixelSize: 11
                            color: Styles.Theme.accentColor
                        }
                    }

                    MouseArea {
                        id: langItemMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: {
                            if (langManager) langManager.load_language(modelData.code)
                            langDialog.close()
                        }
                    }
                }
            }

            Item { Layout.preferredHeight: Styles.Theme.paddingSmall }
        }
    }
}
