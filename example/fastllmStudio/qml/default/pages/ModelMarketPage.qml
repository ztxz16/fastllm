import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../styles" as Styles

Rectangle {
    id: marketPage
    color: Styles.Theme.bgEditor

    signal backClicked()
    signal modelDownloaded()

    property var recommendedModels: []
    property var searchResults: []

    Component.onCompleted: {
        if (marketViewModel) recommendedModels = marketViewModel.getRecommendedModels()
    }

    Connections {
        target: marketViewModel
        function onSearchResultsChanged() { searchResults = marketViewModel.getSearchResults() }
        function onMarketModelsChanged() { recommendedModels = marketViewModel.getRecommendedModels(); searchResults = marketViewModel.getSearchResults() }
        function onDownloadFinished(repoId, localPath) { marketPage.modelDownloaded() }
    }

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

                // Back
                Rectangle {
                    width: backRow.implicitWidth + 20; height: 32; radius: Styles.Theme.borderRadius
                    color: backMa.containsMouse ? Styles.Theme.bgInput : "transparent"
                    Row {
                        id: backRow; anchors.centerIn: parent; spacing: 6
                        Text { text: "←"; font.pixelSize: 15; color: Styles.Theme.textPrimary }
                        Text { text: qsTr("Back"); font.pixelSize: Styles.Theme.fontSizeMedium; color: Styles.Theme.textPrimary }
                    }
                    MouseArea { id: backMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; onClicked: marketPage.backClicked() }
                }

                Text {
                    text: qsTr("Model Market")
                    font.pixelSize: Styles.Theme.fontSizeLarge; font.weight: Font.DemiBold; color: Styles.Theme.textBright
                }

                Item { Layout.fillWidth: true }
            }

            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Styles.Theme.border }
        }

        // Search bar
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 44
            color: Styles.Theme.bgBase

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: Styles.Theme.paddingMedium
                anchors.rightMargin: Styles.Theme.paddingMedium
                spacing: Styles.Theme.spacingSmall

                Rectangle {
                    Layout.fillWidth: true; height: 30; radius: Styles.Theme.borderRadius
                    color: Styles.Theme.bgInput; border.color: searchFld.activeFocus ? Styles.Theme.borderFocus : Styles.Theme.border; border.width: 1

                    TextInput {
                        id: searchFld
                        anchors.fill: parent; anchors.leftMargin: 10; anchors.rightMargin: 10
                        verticalAlignment: TextInput.AlignVCenter
                        color: Styles.Theme.textPrimary; selectionColor: Styles.Theme.accentColor
                        font.pixelSize: Styles.Theme.fontSizeMedium
                        clip: true
                        onAccepted: { if (marketViewModel && text.trim() !== "") marketViewModel.searchModels(text.trim()) }

                        Text {
                            anchors.fill: parent; verticalAlignment: Text.AlignVCenter
                            text: qsTr("Search models...")
                            color: Styles.Theme.textDisabled; font.pixelSize: Styles.Theme.fontSizeMedium
                            visible: !searchFld.text && !searchFld.activeFocus
                        }
                    }
                }

                Rectangle {
                    width: searchLbl.implicitWidth + 20; height: 30; radius: Styles.Theme.borderRadius
                    color: searchMa.containsMouse ? Styles.Theme.primaryHover : Styles.Theme.primaryColor
                    Text { id: searchLbl; anchors.centerIn: parent; text: qsTr("Search"); font.pixelSize: Styles.Theme.fontSizeSmall; color: Styles.Theme.textOnAccent }
                    MouseArea { id: searchMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor
                        onClicked: { if (marketViewModel && searchFld.text.trim() !== "") marketViewModel.searchModels(searchFld.text.trim()) }
                    }
                }
            }

            Rectangle { anchors.bottom: parent.bottom; width: parent.width; height: 1; color: Styles.Theme.border }
        }

        // Section label
        Rectangle {
            Layout.fillWidth: true; Layout.preferredHeight: 30; color: "transparent"
            Text {
                anchors.verticalCenter: parent.verticalCenter; anchors.left: parent.left; anchors.leftMargin: Styles.Theme.paddingMedium
                text: searchResults.length > 0 ? qsTr("Search") + " (" + searchResults.length + ")" : qsTr("Recommended Models")
                font.pixelSize: Styles.Theme.fontSizeSmall; font.weight: Font.DemiBold; color: Styles.Theme.textSecondary; font.capitalization: Font.AllUppercase
            }
        }

        // Model list
        ListView {
            id: marketList
            Layout.fillWidth: true; Layout.fillHeight: true
            Layout.leftMargin: Styles.Theme.paddingMedium; Layout.rightMargin: Styles.Theme.paddingMedium
            spacing: Styles.Theme.spacingSmall; clip: true
            model: searchResults.length > 0 ? searchResults : recommendedModels

            ScrollBar.vertical: ScrollBar {
                policy: ScrollBar.AsNeeded
                contentItem: Rectangle { implicitWidth: 6; radius: 3; color: Styles.Theme.scrollbar }
            }

            delegate: Rectangle {
                width: marketList.width - 2 * Styles.Theme.paddingMedium
                anchors.horizontalCenter: parent ? parent.horizontalCenter : undefined
                height: 64; radius: Styles.Theme.borderRadius
                color: mItemMa.containsMouse ? Styles.Theme.bgCardHover : Styles.Theme.bgCard
                border.color: Styles.Theme.border; border.width: 1

                Behavior on color { ColorAnimation { duration: Styles.Theme.animDuration } }

                MouseArea { id: mItemMa; anchors.fill: parent; hoverEnabled: true; acceptedButtons: Qt.NoButton }

                RowLayout {
                    anchors.fill: parent; anchors.margins: Styles.Theme.paddingMedium; spacing: Styles.Theme.spacing

                    ColumnLayout {
                        Layout.fillWidth: true; spacing: 2
                        Text { text: modelData.name || ""; font.pixelSize: Styles.Theme.fontSizeLarge; font.weight: Font.DemiBold; color: Styles.Theme.textBright; elide: Text.ElideRight; Layout.fillWidth: true }
                        RowLayout {
                            spacing: Styles.Theme.spacingSmall
                            Text { text: modelData.repo_id || ""; font.pixelSize: Styles.Theme.fontSizeTiny; color: Styles.Theme.textSecondary; elide: Text.ElideRight; Layout.fillWidth: true }
                            Rectangle {
                                visible: (modelData.size || "") !== ""
                                width: sizeLbl.implicitWidth + 10; height: 16; radius: 8
                                color: Styles.Theme.badgeBg
                                Text { id: sizeLbl; anchors.centerIn: parent; text: modelData.size || ""; font.pixelSize: Styles.Theme.fontSizeTiny; color: Styles.Theme.badgeText }
                            }
                        }
                    }

                    Rectangle {
                        width: dlBtnLbl.implicitWidth + 24; height: 28; radius: Styles.Theme.borderRadius
                        color: modelData.downloading ? Styles.Theme.bgInput
                             : dlBtnMa.containsMouse ? Styles.Theme.primaryHover : Styles.Theme.primaryColor
                        Text { id: dlBtnLbl; anchors.centerIn: parent; text: modelData.downloading ? qsTr("Downloading...") : qsTr("Download"); font.pixelSize: Styles.Theme.fontSizeSmall; color: Styles.Theme.textOnAccent }
                        MouseArea {
                            id: dlBtnMa; anchors.fill: parent; hoverEnabled: true; cursorShape: Qt.PointingHandCursor; enabled: !modelData.downloading
                            onClicked: { if (marketViewModel) marketViewModel.downloadModel(modelData.repo_id, modelData.name) }
                        }
                    }
                }
            }
        }
    }
}
