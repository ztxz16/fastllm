import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "styles" as Styles
import "pages"

ApplicationWindow {
    id: root
    visible: true
    width: 1200
    height: 780
    minimumWidth: 860
    minimumHeight: 520
    title: " "
    color: Styles.Theme.bgBase

    // ── Activity Bar (narrow icon strip) + Sidebar + Content ──
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ── Activity Bar ──
        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: Styles.Theme.sidebarWidth
            color: Styles.Theme.sidebarBg

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                // Nav icons
                Repeater {
                    model: ListModel {
                        ListElement { icon: "📦"; pageIdx: 0; tip: QT_TR_NOOP("Model Repository") }
                        ListElement { icon: "💬"; pageIdx: 1; tip: QT_TR_NOOP("Local Chat") }
                    }
                    delegate: Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: Styles.Theme.sidebarWidth
                        color: pageStack.currentIndex === model.pageIdx
                              ? Styles.Theme.sidebarItemActive
                              : actBarMa.containsMouse
                                ? Styles.Theme.sidebarItemHover
                                : "transparent"

                        // Active indicator bar
                        Rectangle {
                            width: 2
                            height: parent.height
                            color: pageStack.currentIndex === model.pageIdx
                                   ? Styles.Theme.sidebarIndicator : "transparent"
                        }

                        Text {
                            anchors.centerIn: parent
                            text: model.icon
                            font.pixelSize: 20
                        }

                        MouseArea {
                            id: actBarMa
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: pageStack.currentIndex = model.pageIdx

                            ToolTip.visible: actBarMa.containsMouse
                            ToolTip.text: qsTr(model.tip)
                            ToolTip.delay: 500
                        }
                    }
                }

                Item { Layout.fillHeight: true }

                // Language selector
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: Styles.Theme.sidebarWidth
                    color: langMa.containsMouse ? Styles.Theme.sidebarItemHover : "transparent"

                    Text {
                        anchors.centerIn: parent
                        text: "🌐"
                        font.pixelSize: 18
                    }

                    MouseArea {
                        id: langMa
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: langDialog.open()
                    }
                }
            }
        }

        // ── Thin separator ──
        Rectangle { Layout.fillHeight: true; Layout.preferredWidth: 1; color: Styles.Theme.border }

        // ── Content area ──
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
                    { code: "zh_CN", label: "简体中文" },
                    { code: "en_US", label: "English" }
                ]

                delegate: Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 44
                    Layout.leftMargin: Styles.Theme.paddingMedium
                    Layout.rightMargin: Styles.Theme.paddingMedium
                    radius: Styles.Theme.borderRadius
                    color: langItemMa.containsMouse ? Styles.Theme.bgCardHover : "transparent"

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
                            text: isCurrent ? "✓" : ""
                            font.pixelSize: 14
                            color: Styles.Theme.accentColor
                        }
                    }

                    MouseArea {
                        id: langItemMa
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
