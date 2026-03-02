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
                        onClicked: langPopup.open()
                    }

                    Popup {
                        id: langPopup
                        x: parent.width + 4
                        y: 0
                        width: 150
                        padding: 4
                        background: Rectangle {
                            color: Styles.Theme.bgPopup
                            border.color: Styles.Theme.border
                            radius: Styles.Theme.borderRadius
                        }
                        contentItem: Column {
                            spacing: 2
                            Repeater {
                                model: langManager ? langManager.availableLanguages() : []
                                delegate: Rectangle {
                                    width: 142
                                    height: 28
                                    radius: 3
                                    color: langItemMa.containsMouse ? Styles.Theme.sidebarItemHover : "transparent"
                                    Text {
                                        anchors.verticalCenter: parent.verticalCenter
                                        anchors.left: parent.left
                                        anchors.leftMargin: 8
                                        text: langManager ? langManager.languageDisplayName(modelData) : modelData
                                        font.pixelSize: Styles.Theme.fontSizeMedium
                                        color: Styles.Theme.textPrimary
                                    }
                                    MouseArea {
                                        id: langItemMa
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: {
                                            if (langManager) langManager.load_language(modelData)
                                            langPopup.close()
                                        }
                                    }
                                }
                            }
                        }
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
}
